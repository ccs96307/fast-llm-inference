import optuna
from typing import Dict, Optional, Tuple

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import time

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from layerskip_modeling.modeling_layerskip_gemma2 import LayerSkipGemma2ForCausalLM
from sampling.sampling import sample_next_token


def calculate_continuous_acceptance(acceptance_mask: torch.BoolTensor) -> int:
    continuous_acceptance = 0
    for accepted in acceptance_mask.long().squeeze(0):
        if accepted == 1:
            continuous_acceptance += 1
        else:
            break
    return continuous_acceptance


def drafter_speculative_decode(
    draft_model: torch.nn.Module,
    draft_tokenizer: PreTrainedTokenizerBase,
    inputs: Dict[str, torch.Tensor],
    gamma: int = 10,
    temperature: float = 1.0,
    top_k: Optional[int] = 0,  # Default is 0, it means do not select top-k tokens
    top_p: Optional[float] = 1.0,
    repetition_penalty: Optional[float] = 1.0,
    draft_mode: bool = True
) -> Tuple[Dict[str, torch.Tensor], torch.FloatTensor]:
    draft_model.set_draft_mode(draft_mode)
    draft_probs = []

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

    draft_model.set_draft_mode(True)

    return inputs, torch.cat(draft_probs, dim=1)


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


def objective(trial):
    # Define search space, sssume we can skip up to six layers
    total_layers = 26

    # Determine skip or not for `attn`
    skip_attn_layers = []
    for i in range(total_layers):
        skip = trial.suggest_int(f'skip_attn_layer_{i}', 0, 1)
        if skip == 1:
            skip_attn_layers.append(i)

    # Determine skip or not for `mlp`
    skip_mlp_layers = []
    for i in range(total_layers):
        skip = trial.suggest_int(f'skip_mlp_layer_{i}', 0, 1)
        if skip == 1:
            skip_mlp_layers.append(i)

    # Disable set to 0 both
    if len(skip_attn_layers) == 0 and len(skip_mlp_layers) == 0:
        raise optuna.TrialPruned()

    skip_layer_ids = {
        "attn": skip_attn_layers,
        "mlp": skip_mlp_layers,
    }

    # Set the skip strategy
    model.set_skip_layer_ids(skip_layer_ids=skip_layer_ids)

    messages = [
        [
            {
                "role": "user",
                "content": "What is the capital of Taiwan. And why?",
            },
        ],
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(device)

    is_end = False

    # Record
    raw_inputs = copy.deepcopy(inputs)
    raw_token_num = raw_inputs["input_ids"].shape[1]

    total_draft_tokens = 0
    total_accept_tokens = 0
    gamma = 5
    max_new_tokens = 100

    while not is_end:
        # Draft model
        target_inputs, draft_probs = drafter_speculative_decode(
            draft_model=model,
            draft_tokenizer=tokenizer,
            inputs=inputs,
            gamma=gamma,
        )

        total_draft_tokens += gamma

        # Target model
        outputs, is_end, accept_tokens = target_speculative_decode(
            target_model=model,
            target_tokenizer=tokenizer,
            inputs=target_inputs,
            draft_probs=draft_probs,
        )

        total_accept_tokens += accept_tokens
        inputs = outputs

        if inputs["input_ids"].shape[1] - raw_token_num >= max_new_tokens:
            break

    # Compute acceptance rate
    accept_rate = total_accept_tokens / total_draft_tokens

    print(f"attn_skip: {skip_attn_layers}, mlp_skip: {skip_mlp_layers}, Accept Rate: {accept_rate}")

    # Assume we want to maximize `accept_rate`
    return accept_rate


if __name__ == "__main__":
    pretrained_model_name_or_path = "../models/google--gemma-2-2b-it/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    model = LayerSkipGemma2ForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16).to(device)

    # Init
    skip_layer_ids = {
        "attn": [],
        "mlp": [],
    }

    model.set_skip_layer_ids(skip_layer_ids=skip_layer_ids)

    # Create
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    print("The best params:", study.best_params)
    print("The best accept_rate:", study.best_value)
