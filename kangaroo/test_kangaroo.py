from typing import Dict, List, Optional, Tuple

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import time

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kangaroo_modeling.modeling_kangaroo_llama3 import KangarooLlamaForCausalLM
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
) -> Tuple[Dict[str, torch.Tensor], torch.FloatTensor]:
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
    with torch.no_grad():
        outputs = target_model(**inputs)

    next_tokens, target_probs = sample_next_token(
        logits=outputs.logits,
        diff_probs=draft_probs,
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


def run_test() -> None:
    # Device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model path 
    pretrained_model_name_or_path = "../models/meta-llama--Meta-Llama-3.1-8B-Instruct"
    adapter_dir = "checkpoints/checkpoints_hce_decoder_layer_20241205/epoch_45/"

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    model = KangarooLlamaForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
    model.set_skip_layer(shallow_layer_num=2)
    model.set_adapter_layer("decoder_layer")
    if adapter_dir:
        model.load_adapter(adapter_dir)

    model = model.to(device)

    # Tokenize
    messages = [
        [
            {
                "role": "user",
                "content": "What is the capital of Taiwan. And why?",
            },
        ],
    ]

    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(device)

    # Warm up the model (CUDA)
    inputs_dummy = {k: v.clone() for k, v in inputs.items()}
    with torch.no_grad():
        model.set_draft_mode()
        model(**inputs_dummy)
        model.set_target_mode()
        model(**inputs_dummy)
    torch.cuda.synchronize()

    # Record
    raw_inputs = copy.deepcopy(inputs)
    raw_token_num = raw_inputs["input_ids"].shape[1]
    total_draft_tokens = 0
    total_accept_tokens = 0
    gamma = 1
    max_new_tokens = 100
    is_end = False

    start_time = time.time()

    while not is_end:
        # Draft model
        model.set_draft_mode()
        target_inputs, draft_probs = drafter_speculative_decode(
            draft_model=model,
            draft_tokenizer=tokenizer,
            inputs=inputs,
            gamma=gamma,
            temperature=0,
        )

        total_draft_tokens += gamma

        # Target model
        model.set_target_mode()
        outputs, is_end, accept_tokens = target_speculative_decode(
            target_model=model,
            target_tokenizer=tokenizer,
            inputs=target_inputs,
            draft_probs=draft_probs,
            temperature=1,
        )

        total_accept_tokens += accept_tokens

        inputs = outputs

        if inputs["input_ids"].shape[1] - raw_token_num >= max_new_tokens:
            break

    generate_token_num = outputs["input_ids"].shape[1] - raw_token_num
    spent_time = time.time() - start_time

    print(f"Generate token number: {generate_token_num}")
    print(f"Generate speed: {generate_token_num / spent_time} tokens/sec")
    print(f"Speculative Decoding Spent Time: {spent_time} seconds.")
    print(f"Accept Rate: {total_accept_tokens / total_draft_tokens}\n")

    # Normal Target Model Speed
    inputs = copy.deepcopy(raw_inputs)
    start_time = time.time()
    target_inputs, draft_probs = drafter_speculative_decode(
        draft_model=model,
        draft_tokenizer=tokenizer,
        inputs=inputs,
        gamma=max_new_tokens,
    )

    spent_time = time.time() - start_time

    print(f"Generate token number: {max_new_tokens}")
    print(f"Generate speed: {max_new_tokens / spent_time} tokens/sec")
    print(f"Normal Target Model Decoding Spent Time: {spent_time} seconds.\n")


if __name__ == "__main__":
    run_test()
