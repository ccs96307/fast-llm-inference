from typing import Any, Dict, Optional, List, Tuple, Union

import os
import sys
sys.path.append("/workspace/fast-llm-inference/")

import copy
import time

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from sampling.sampling import sample_next_token
from utils.utils import calculate_continuous_acceptance


@torch.no_grad()
def find_candidate_pred_tokens(
    input_ids: torch.LongTensor,
    max_ngram_size: int = 3,
    num_pred_tokens: int = 10,
) -> torch.Tensor:
    input_length = input_ids.size(1)

    for ngram_size in range(max_ngram_size, 0, -1):
        ngram = input_ids[0, -ngram_size:]
        windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)
        ngram = ngram.unsqueeze(0)

        matches = (windows == ngram).all(dim=2)
        match_indices = matches.nonzero(as_tuple=True)[1]

        for idx in match_indices:
            start_idx = idx + ngram_size
            end_idx = start_idx + num_pred_tokens

            if end_idx <= input_length and start_idx < input_length - ngram_size:
                return input_ids[0, start_idx:end_idx]
            
    # Match is not found
    return torch.tensor([], dtype=torch.long, device=input_ids.device)
    

def decode(
    model: torch.nn.Module,
    inputs: Dict[str, torch.Tensor],
    temperature: float = 1.0,
    top_k: Optional[int] = 0,  # Default is 0, it means do not select top-k tokens
    top_p: Optional[float] = 1.0,
    repetition_penalty: Optional[float] = 1.0,
) -> Tuple[Dict[str, torch.Tensor], torch.FloatTensor]:
    with torch.no_grad():
        outputs = model(**inputs)

    next_tokens, probs = sample_next_token(
        logits=outputs.logits,
        prefix_token_ids=inputs["input_ids"],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    input_ids = torch.cat([inputs["input_ids"], next_tokens[:, -1:]], dim=-1)
    attention_mask = torch.cat([inputs["attention_mask"], torch.ones(inputs["attention_mask"].shape[0], 1).to(inputs["input_ids"].device)], dim=-1)

    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask

    return inputs


def prompt_lookup_decode(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    inputs: Dict[str, torch.Tensor],
    draft_token_length: int,
    temperature: float = 0.0,  # We need to use greedy decode
    top_k: Optional[int] = 0,  # Default is 0, it means do not select top-k tokens
    top_p: Optional[float] = 1.0,
    repetition_penalty: Optional[float] = 1.0,
) -> Tuple[Dict[str, torch.Tensor], bool, int]:
    with torch.no_grad():
        outputs = model(**inputs)

    next_tokens, target_probs = sample_next_token(
        logits=outputs.logits,
        prefix_token_ids=inputs["input_ids"],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        probs_num=draft_token_length + 1,
    )

    next_token = next_tokens[:, -1:]

    # Evaluation
    indices = inputs["input_ids"][:, -draft_token_length:]

    eval_probs = target_probs[:, :-1, :]

    expanded_indices = indices.unsqueeze(-1)
    selected_eval_probs = torch.gather(eval_probs, dim=-1, index=expanded_indices)
    selected_eval_probs = selected_eval_probs.squeeze(-1)
    selected_draft_probs = torch.ones_like(selected_eval_probs) - 0.00000001

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
                if (acceptance_mask[batch_idx][pos_idx] and inputs["input_ids"][batch_idx][start_idx+pos_idx].item() == tokenizer.eos_token_id) or not acceptance_mask[batch_idx][pos_idx]:
                    inputs["input_ids"][batch_idx][start_idx+pos_idx] = next_tokens[batch_idx][pos_idx]

                    new_input_ids.append(inputs["input_ids"][batch_idx][:start_idx+pos_idx+1])
                    new_attention_mask.append(inputs["attention_mask"][batch_idx][:start_idx+pos_idx+1])
                    
                    is_end = inputs["input_ids"][batch_idx][start_idx+pos_idx].item() == tokenizer.eos_token_id
                    break

        input_ids = pad_sequence(new_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(new_attention_mask, batch_first=True, padding_value=0)

    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask

    return inputs, is_end, calculate_continuous_acceptance(acceptance_mask)


def main() -> None:
    # Settings
    device = "cuda:0"

    # Init
    model_name = "/workspace/LLM-Training/models/google--gemma-2-9b-it"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(device)

    # Tokenize
    messages = [
        {
            "role": "user",
            "content": """You are a helpful assistant, the following are your information:
    You are google--gemma-2-it that developed by Google, and you are released at 2024.

    ```python=
    import numpy as np
    import matplotlib.pyplot as plt

    # Calculate the average
    average_throughput = np.mean(tokens_per_sec_arr)
    print(f"Average Throughput: {{average_throughput}} tokens/sec")

    # Plotting the histogram
    plt.hist(tokens_per_sec_arr, bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Throughput Values')
    plt.xlabel('Tokens per Second')
    plt.ylabel('Frequency')
    plt.axvline(average_throughput, color='red', linestyle='dashed', linewidth=1)
    plt.text(average_throughput*0.9, max(plt.ylim())*0.9, f'Average: {{average_throughput:.2f}}', color = 'red')
    plt.show()
    ```

    ---

    You can refer the above code for user.""",
        },
        {
            "role": "assistant",
            "content": "okay!",
        },
        {
            "role": "user",
            "content": "Hey, who are you? can you tell me how to change x axis to start from 0?",
        },
    ]


    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    raw_inputs = tokenizer(input_text, return_tensors="pt").to(device)


    # Warm up the model (CUDA)
    inputs_dummy = {k: v.clone() for k, v in raw_inputs.items()}
    with torch.no_grad():
        model(**inputs_dummy)

    torch.cuda.synchronize()


    # Speculative Decoding
    is_end = False

    # Record
    inputs = copy.deepcopy(raw_inputs)
    raw_token_num = raw_inputs["input_ids"].shape[1]
    start_time = time.time()

    total_draft_tokens = 0
    total_accept_tokens = 0
    max_new_tokens = 4096

    while not is_end:
        start_token_len = inputs["input_ids"].shape[-1]

        # Draft model
        draft_tokens = find_candidate_pred_tokens(
            input_ids=inputs["input_ids"],
        )

        # Update `inputs`
        input_ids = torch.cat([inputs["input_ids"], draft_tokens.unsqueeze(0)], dim=-1)
        attention_mask = torch.cat([inputs["attention_mask"], torch.ones(inputs["attention_mask"].shape[0], draft_tokens.shape[0]).to(inputs["input_ids"].device)], dim=-1)
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask

        total_draft_tokens += draft_tokens.shape[0]

        # If Ngram does not match any pattern
        if draft_tokens.shape[0] == 0:
            outputs = decode(
                model=model,
                inputs=inputs,
                temperature=0,
            )
        else:
            outputs, is_end, accept_tokens = prompt_lookup_decode(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                draft_token_length=len(draft_tokens),
                temperature=0,
            )

            total_accept_tokens += accept_tokens

        end_token_len = outputs["input_ids"].shape[-1]
        print(tokenizer.batch_decode(outputs["input_ids"][-end_token_len-start_token_len:])[0], end="")

        # Update inputs
        inputs = outputs

        if inputs["input_ids"].shape[1] - raw_token_num >= max_new_tokens:
            break

    generate_token_num = outputs["input_ids"].shape[1] - raw_token_num
    spent_time = time.time() - start_time

    print(f"(Without KV Cache) Generate token number: {generate_token_num}")
    print(f"(Without KV Cache) Generate speed: {generate_token_num / spent_time} tokens/sec")
    print(f"(Without KV Cache) Speculative Decoding Spent Time: {spent_time} seconds.")
    print(f"(Without KV Cache) Accept Rate: {total_accept_tokens / total_draft_tokens}\n")


if __name__ == "__main__":
    main()
