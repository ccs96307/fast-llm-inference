from typing import Dict, List, Optional, Tuple, Union

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import copy
import time

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, Cache, DynamicCache, PreTrainedTokenizerBase

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
        raw_inputs_ids = inputs.input_ids

        if isinstance(past_key_values, Cache) and past_key_values.get_seq_length() > 0:
            distance = inputs.input_ids.shape[1] - past_key_values.get_seq_length()

            if distance >= 1:
                inputs.input_ids = inputs.input_ids[:, -distance:]
            else:
                past_key_values.crop(max_length=inputs.input_ids.shape[1]-1)
                inputs.input_ids = inputs.input_ids[:, -1:]

        with torch.no_grad():
            outputs = draft_model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                past_key_values=past_key_values,
                use_cache=past_key_values is not None,
            )

        past_key_values = outputs.past_key_values

        next_tokens, probs = sample_next_token(
            logits=outputs.logits,
            prefix_token_ids=inputs.input_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        draft_probs.append(probs)
        input_ids = torch.cat([raw_inputs_ids, next_tokens[:, -1:]], dim=-1)
        attention_mask = torch.cat([inputs.attention_mask, torch.ones(inputs.attention_mask.shape[0], 1).to(inputs.input_ids.device)], dim=-1)

        inputs.input_ids = input_ids
        inputs.attention_mask = attention_mask

    return inputs, torch.cat(draft_probs, dim=1), past_key_values


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
    raw_inputs_ids = inputs.input_ids

    if isinstance(past_key_values, Cache) and past_key_values.get_seq_length() > 0:
        distance = inputs.input_ids.shape[1] - past_key_values.get_seq_length()
        if distance >= 1:
            inputs.input_ids = inputs.input_ids[:, -distance:]
        else:
            past_key_values.crop(max_length=inputs.input_ids.shape[1]-1)
            inputs.input_ids = inputs.input_ids[:, -1:]

    with torch.no_grad():
        outputs = target_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            past_key_values=past_key_values,
            use_cache=past_key_values is not None,
        )

    past_key_values = outputs.past_key_values
    inputs.input_ids = raw_inputs_ids

    next_tokens, target_probs = sample_next_token(
        logits=outputs.logits,
        diff_probs=draft_probs,
        prefix_token_ids=inputs.input_ids,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        probs_num=draft_probs.shape[1] + 1,
    )

    next_token = next_tokens[:, -1:]

    # Evaluation
    indices = inputs.input_ids[:, -draft_probs.shape[1]:]

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
        inputs.input_ids = torch.cat([inputs.input_ids, next_token], dim=-1)
        inputs.attention_mask = torch.cat([inputs.attention_mask, torch.ones(inputs.attention_mask.shape[0], 1).to(inputs.input_ids.device)], dim=-1)
    else:
        new_input_ids = []
        new_attention_mask = []

        for batch_idx in range(next_tokens.shape[0]):
            gamma = next_tokens.shape[1] - 1
            start_idx = inputs.input_ids.shape[1] - gamma

            for pos_idx in range(acceptance_mask[batch_idx].shape[0]):
                if (acceptance_mask[batch_idx][pos_idx] and inputs.input_ids[batch_idx][start_idx+pos_idx].item() == target_tokenizer.eos_token_id) or not acceptance_mask[batch_idx][pos_idx]:
                    inputs.input_ids[batch_idx][start_idx+pos_idx] = next_tokens[batch_idx][pos_idx]

                    new_input_ids.append(inputs.input_ids[batch_idx][:start_idx+pos_idx+1])
                    new_attention_mask.append(inputs.attention_mask[batch_idx][:start_idx+pos_idx+1])
                    
                    is_end = inputs.input_ids[batch_idx][start_idx+pos_idx].item() == target_tokenizer.eos_token_id
                    break

        input_ids = pad_sequence(new_input_ids, batch_first=True, padding_value=target_tokenizer.pad_token_id)
        attention_mask = pad_sequence(new_attention_mask, batch_first=True, padding_value=0)

        inputs.input_ids = input_ids
        inputs.attention_mask = attention_mask

    if isinstance(past_key_values, Cache) and inputs.input_ids.shape[1] <= past_key_values.get_seq_length():
        past_key_values.crop(max_length=inputs.input_ids.shape[1]-1)

    return inputs, is_end, calculate_continuous_acceptance(acceptance_mask), past_key_values


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
    ]

    input_text=draft_tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = draft_tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(device)

    # Warm up the model (CUDA)
    inputs_dummy = {k: v.clone() for k, v in inputs.items()}
    with torch.no_grad():
        draft_model(**inputs_dummy)
        target_model(**inputs_dummy)
    torch.cuda.synchronize()

    # Speculative Decoding
    is_end = False

    # Record
    raw_inputs = copy.deepcopy(inputs)
    raw_token_num = raw_inputs.input_ids.shape[1]
    start_time = time.time()

    total_draft_tokens = 0
    total_accept_tokens = 0
    gamma = args.gamma
    max_new_tokens = args.test_token_num

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

        total_draft_tokens += gamma

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

        if inputs.input_ids.shape[1] - raw_token_num >= max_new_tokens:
            break

    generate_token_num = outputs.input_ids.shape[1] - raw_token_num
    spent_time = time.time() - start_time

    print(draft_tokenizer.batch_decode(inputs.input_ids)[0])

    print(f"(Without KV Cache) Generate token number: {generate_token_num}")
    print(f"(Without KV Cache) Generate speed: {generate_token_num / spent_time} tokens/sec")
    print(f"(Without KV Cache) Speculative Decoding Spent Time: {spent_time} seconds.")
    print(f"(Without KV Cache) Accept Rate: {total_accept_tokens / total_draft_tokens}\n")


    # KV Cache Speculative Decoding
    is_end = False

    # Record
    inputs = copy.deepcopy(raw_inputs)
    raw_token_num = inputs.input_ids.shape[1]
    start_time = time.time()

    total_draft_tokens = 0
    total_accept_tokens = 0
    gamma = args.gamma
    max_new_tokens = args.test_token_num

    draft_past_key_values = DynamicCache()
    target_past_key_values = DynamicCache()

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

        total_draft_tokens += gamma

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

        if inputs.input_ids.shape[1] - raw_token_num >= max_new_tokens:
            break

    generate_token_num = outputs.input_ids.shape[1] - raw_token_num
    spent_time = time.time() - start_time

    print(draft_tokenizer.batch_decode(inputs.input_ids)[0])

    print(f"(KV Cache) Generate token number: {generate_token_num}")
    print(f"(KV Cache) Generate speed: {generate_token_num / spent_time} tokens/sec")
    print(f"(KV Cache) Speculative Decoding Spent Time: {spent_time} seconds.")
    print(f"(KV Cache) Accept Rate: {total_accept_tokens / total_draft_tokens}\n")

    # Normal Target Model Speed
    inputs = copy.deepcopy(raw_inputs)
    past_key_values = DynamicCache()
    start_time = time.time()
    target_inputs, draft_probs, _ = drafter_speculative_decode(
        draft_model=target_model,
        draft_tokenizer=draft_tokenizer,
        inputs=inputs,
        gamma=args.test_token_num,
        temperature=0,
        past_key_values=past_key_values,
    )

    spent_time = time.time() - start_time

    # print(draft_tokenizer.batch_decode(target_inputs.input_ids)[0])

    print(f"Generate token number: {max_new_tokens}")
    print(f"Generate speed: {max_new_tokens / spent_time} tokens/sec")
    print(f"Normal Target Model Decoding Spent Time: {spent_time} seconds.\n")

    # Normal Draft Model Speed
    inputs = copy.deepcopy(raw_inputs)
    past_key_values = DynamicCache()
    start_time = time.time()
    target_inputs, draft_probs, _ = drafter_speculative_decode(
        draft_model=draft_model,
        draft_tokenizer=draft_tokenizer,
        inputs=inputs,
        gamma=args.test_token_num,
        past_key_values=past_key_values,
    )

    spent_time = time.time() - start_time

    print(f"Generate token number: {max_new_tokens}")
    print(f"Generate speed: {max_new_tokens / spent_time} tokens/sec")
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
