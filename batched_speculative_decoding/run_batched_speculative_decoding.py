from typing import Dict, List, Optional, Tuple, Union

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import copy
import time

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, Cache, PreTrainedTokenizerBase

from sampling.sampling import sample_next_token


"""
python speculative_decoding/run_speculative_decoding.py \
    --target_model_path HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --draft_model_path HuggingFaceTB/SmolLM2-135M-Instruct \
    --device cuda:0 \
    --question 'What is the capital of Taiwan. And why?' \
    --gamma 5 \
    --test_token_num 100
"""


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
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.FloatTensor, Optional[Union[Cache, List[torch.FloatTensor]]]]:
    draft_probs = []

    for idx in range(gamma):
        with torch.no_grad():
            outputs = draft_model(
                **inputs,
                past_key_values = past_key_values,
                use_cache = past_key_values is not None,
            )

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

    return inputs, torch.cat(draft_probs, dim=1), outputs.past_key_values


def target_speculative_decode(
    target_model: torch.nn.Module,
    target_tokenizer: PreTrainedTokenizerBase,
    inputs: Dict[str, torch.Tensor],
    draft_probs: torch.FloatTensor,
    temperature: float = 1.0,
    top_k: Optional[int] = 0,  # Default is 0, it means do not select top-k tokens
    top_p: Optional[float] = 1.0,
    repetition_penalty: Optional[float] = 1.0,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
) -> Tuple[Dict[str, torch.Tensor], bool, int, Optional[Union[Cache, List[torch.FloatTensor]]]]:
    with torch.no_grad():
        outputs = target_model(
            **inputs,
            past_key_values = past_key_values,
            use_cache = past_key_values is not None,
        )

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
    gamma = next_tokens.shape[1] - 1
    start_idx = inputs.input_ids.shape[1] - gamma
    acceptance_mask = (acceptance_mask == False).cumsum(dim=-1) <= 0

    inputs.attention_mask[:, start_idx:] = acceptance_mask.float()

    false_indices = torch.where(
        ~acceptance_mask,
        torch.arange(acceptance_mask.shape[-1], device=acceptance_mask.device).expand_as(acceptance_mask),
        acceptance_mask.shape[1],
    )

    selected_indices = torch.where(
        false_indices.min(dim=-1).values >= acceptance_mask.shape[-1],
        next_tokens.shape[-1] - 1,
        false_indices.min(dim=-1).values,
    )

    next_token = next_tokens[torch.arange(next_tokens.shape[0], device=next_tokens.device), selected_indices].unsqueeze(1)
    next_attn_mask_value = torch.ones(next_token.shape[0], 1, device=next_token.device)

    input_ids = torch.cat([inputs.input_ids, next_token], dim=-1)
    attention_mask = torch.cat([inputs.attention_mask, next_attn_mask_value], dim=-1)         

    inputs.input_ids = input_ids
    inputs.attention_mask = attention_mask

    is_end = False

    return inputs, is_end, acceptance_mask.sum().item(), outputs.past_key_values


def run_test(args) -> None:
    # Device
    device = torch.device(args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model path 
    target_model_path = args.target_model_path
    draft_model_path = args.draft_model_path

    # Load Tokenizer
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_path)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path)

    # Load Model
    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_path, torch_dtype=torch.bfloat16).to(device)
    target_model = AutoModelForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.bfloat16).to(device)

    # Tokenize
    messages = [
        [
            {
                "role": "user",
                "content": args.question,
            },
        ],
        # [
        #     {
        #         "role": "user",
        #         "content": "Who are you?",
        #     },
        # ],
    ]

    input_text=draft_tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = draft_tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(device)

    batch_size = inputs.input_ids.shape[0]

    # Warm up the model (CUDA)
    inputs_dummy = {k: v.clone() for k, v in inputs.items()}
    with torch.no_grad():
        draft_model(**inputs_dummy)
        target_model(**inputs_dummy)
    torch.cuda.synchronize()

    is_end = False

    # Record
    raw_inputs = copy.deepcopy(inputs)
    start_time = time.time()

    total_draft_tokens = 0
    total_accept_tokens = 0
    gamma = args.gamma
    max_new_tokens = args.test_token_num

    raw_attention_mask_num = inputs.attention_mask.clone().sum(-1)

    draft_past_key_values = None
    target_past_key_values = None

    while not is_end:
        # Draft model
        target_inputs, draft_probs, draft_past_key_values = drafter_speculative_decode(
            draft_model=draft_model,
            draft_tokenizer=draft_tokenizer,
            inputs=inputs,
            gamma=gamma,
            temperature=0,
            past_key_values=draft_past_key_values,
        )

        total_draft_tokens += batch_size * gamma

        # Target model
        outputs, is_end, accept_tokens, target_past_key_values = target_speculative_decode(
            target_model=target_model,
            target_tokenizer=target_tokenizer,
            inputs=target_inputs,
            draft_probs=draft_probs,
            temperature=0,
            past_key_values=target_past_key_values,
        )

        total_accept_tokens += accept_tokens

        inputs = outputs

        attention_mask_num = inputs.attention_mask.sum(-1)
        if (attention_mask_num - raw_attention_mask_num).min().item() >= max_new_tokens:
            break

    generate_token_num = (attention_mask_num - raw_attention_mask_num).sum().item()
    spent_time = time.time() - start_time

    # print(draft_tokenizer.batch_decode(outputs["input_ids"])[0])

    print(f"Generate token number: {generate_token_num}")
    print(f"Generate speed: {generate_token_num / spent_time} tokens/sec")
    print(f"Speculative Decoding Spent Time: {spent_time} seconds.")
    print(f"Accept Rate: {total_accept_tokens / total_draft_tokens}\n")

    # Normal Target Model Speed
    inputs = copy.deepcopy(raw_inputs)
    start_time = time.time()
    target_inputs, draft_probs, _ = drafter_speculative_decode(
        draft_model=target_model,
        draft_tokenizer=draft_tokenizer,
        inputs=inputs,
        gamma=args.test_token_num,
        temperature=0,
    )

    spent_time = time.time() - start_time

    # print(draft_tokenizer.batch_decode(target_inputs["input_ids"])[0])

    print(f"Generate token number: {batch_size * max_new_tokens}")
    print(f"Generate speed: {batch_size * max_new_tokens / spent_time} tokens/sec")
    print(f"Normal Target Model Decoding Spent Time: {spent_time} seconds.\n")

    # Normal Draft Model Speed
    inputs = copy.deepcopy(raw_inputs)
    start_time = time.time()
    target_inputs, draft_probs, _ = drafter_speculative_decode(
        draft_model=draft_model,
        draft_tokenizer=draft_tokenizer,
        inputs=inputs,
        gamma=args.test_token_num,
    )

    spent_time = time.time() - start_time

    print(f"Generate token number: {batch_size * max_new_tokens}")
    print(f"Generate speed: {batch_size * max_new_tokens / spent_time} tokens/sec")
    print(f"Normal Draft Model Decoding Spent Time: {spent_time} seconds.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model_path", type=str, default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--draft_model_path", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--question", type=str, default="What is the capital of Taiwan. And why?")
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--test_token_num", type=int, default=100)
    args = parser.parse_args()

    run_test(args)
