import optuna
from typing import Dict, Optional, Tuple

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import time

from datasets import load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from layerskip_modeling.modeling_layerskip_gemma2 import LayerSkipGemma2ForCausalLM
from sampling.sampling import sample_next_token
from utils.utils import calculate_continuous_acceptance, AdaptiveDraftExitAduster
from utils.optimization_searcher import LayerSkipStrategySearcher


def drafter_speculative_decode(
    draft_model: torch.nn.Module,
    draft_tokenizer: PreTrainedTokenizerBase,
    inputs: Dict[str, torch.Tensor],
    gamma: int = 10,
    temperature: float = 1.0,
    top_k: Optional[int] = 0,  # Default is 0, it means do not select top-k tokens
    top_p: Optional[float] = 1.0,
    repetition_penalty: Optional[float] = 1.0,
    draft_mode: bool = True,
    confidence_threshold_adjuster: Optional[AdaptiveDraftExitAduster] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.FloatTensor]:
    draft_model.set_draft_mode(draft_mode)
    draft_probs = []
    real_generated_tokens = 0

    for idx in range(gamma):
        with torch.no_grad():
            outputs = draft_model(**inputs)

        next_tokens, probs = sample_next_token(
            logits=outputs.logits,
            prefix_token_ids=inputs["input_ids"],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        draft_probs.append(probs)
        input_ids = torch.cat([inputs["input_ids"], next_tokens[:, -1:]], dim=-1)
        attention_mask = torch.cat([inputs["attention_mask"], torch.ones(inputs["attention_mask"].shape[0], 1).to(inputs["input_ids"].device)], dim=-1)

        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask

        real_generated_tokens += 1

        # Early exit
        if confidence_threshold_adjuster and confidence_threshold_adjuster.should_exit(draft_prob=probs[0, 0, next_tokens.item()]):
            # print(confidence_threshold_adjuster.get_state())
            break

    draft_model.set_draft_mode(False)

    return inputs, torch.cat(draft_probs, dim=1), real_generated_tokens


def target_speculative_decode(
    target_model: torch.nn.Module,
    target_tokenizer: PreTrainedTokenizerBase,
    inputs: Dict[str, torch.Tensor],
    draft_probs: torch.FloatTensor,
    temperature: float = 1.0,
    top_k: Optional[int] = 0,  # Default is 0, it means do not select top-k tokens
    top_p: Optional[float] = 1.0,
    repetition_penalty: Optional[float] = 1.0,
) -> Tuple[Dict[str, torch.Tensor], bool, int]:
    target_model.set_draft_mode(False)
    with torch.no_grad():
        outputs = target_model(**inputs)

    next_tokens, target_probs = sample_next_token(
        logits=outputs.logits,
        prefix_token_ids=inputs["input_ids"],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        probs_num=draft_probs.shape[1] + 1,
    )

    next_token = next_tokens[:, -1:]

    # Evaluation
    indices = inputs["input_ids"][:, -draft_probs.shape[1]:]

    eval_probs = target_probs[:, :-1, :]

    expanded_indices = indices.unsqueeze(-1)
    selected_draft_probs = torch.gather(draft_probs, dim=-1, index=expanded_indices)
    selected_draft_probs = selected_draft_probs.squeeze(-1)

    selected_eval_probs = torch.gather(eval_probs, dim=-1, index=expanded_indices)
    selected_eval_probs = selected_eval_probs.squeeze(-1)

    # Compare draft_prob and eval_prob, and check the reject_mask
    mask_to_reject = selected_draft_probs > selected_eval_probs

    # Calculate reject probabilty 1 - (eval_prob / draft_prob)
    rejection_probs = 1 - (selected_eval_probs / selected_draft_probs)

    # Generate random values to determined accept or reject
    random_values = torch.rand_like(rejection_probs)
    rejection_decisions = random_values < rejection_probs

    # Get the final reject masks
    rejection_masks = mask_to_reject & rejection_decisions
    acceptance_mask = torch.ones_like(selected_draft_probs, dtype=torch.bool)
    acceptance_mask[rejection_masks] = False

    is_end = False

    # Concat `input_ids`
    if torch.all(acceptance_mask):
        input_ids = torch.cat([inputs["input_ids"], next_token], dim=-1)
        attention_mask = torch.cat([inputs["attention_mask"], torch.ones(inputs["attention_mask"].shape[0], 1).to(inputs["input_ids"].device)], dim=-1)
    else:
        new_input_ids = []
        new_attention_mask = []

        for batch_idx in range(next_tokens.shape[0]):
            gamma = next_tokens.shape[1] - 1
            start_idx = inputs["input_ids"].shape[1] - gamma

            for pos_idx in range(acceptance_mask[batch_idx].shape[0]):
                if (acceptance_mask[batch_idx][pos_idx] and inputs["input_ids"][batch_idx][start_idx+pos_idx].item() == target_tokenizer.eos_token_id) or not acceptance_mask[batch_idx][pos_idx]:
                    inputs["input_ids"][batch_idx][start_idx+pos_idx] = next_tokens[batch_idx][pos_idx]

                    new_input_ids.append(inputs["input_ids"][batch_idx][:start_idx+pos_idx+1])
                    new_attention_mask.append(inputs["attention_mask"][batch_idx][:start_idx+pos_idx+1])
                    
                    is_end = inputs["input_ids"][batch_idx][start_idx+pos_idx].item() == target_tokenizer.eos_token_id
                    break

        input_ids = pad_sequence(new_input_ids, batch_first=True, padding_value=target_tokenizer.pad_token_id)
        attention_mask = pad_sequence(new_attention_mask, batch_first=True, padding_value=0)

    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask

    return inputs, is_end, calculate_continuous_acceptance(acceptance_mask)


if __name__ == "__main__":
    # Load tokenizer and model
    pretrained_model_name_or_path = "../models/google--gemma-2-2b-it/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    model = LayerSkipGemma2ForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16).to(device)

    # Load dataset
    dataset = load_dataset("shibing624/sharegpt_gpt4")
    samples = dataset["train"]["conversations"][:10]
    samples = [[{"role": sample[0]["from"].replace("human", "user").replace("gpt", "assistant"), "content": sample[0]["value"]}] for sample in samples]

    # Adaptive draft threshold adjuster
    confidence_threshold_adjuster = AdaptiveDraftExitAduster(
        target_matchness=0.5,
        beta1=0.5,
        beta2=0.9,
        epsilon=0.1,
        max_step_draft=8,
    )

    # Searcher
    searcher = LayerSkipStrategySearcher(
        model=model,
        tokenizer=tokenizer,
        device=device,
        drafter_speculative_decode=drafter_speculative_decode,
        target_speculative_decode=target_speculative_decode,
        adjuster=confidence_threshold_adjuster,
    )

    searcher.optimize_speculative_speed(
        samples=samples,
        n_trials=100,
    )
