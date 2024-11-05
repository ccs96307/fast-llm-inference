from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from transformers import LlamaForCausalLM, GPT2TokenizerFast, PreTrainedTokenizerBase

from sampling import sample_next_token


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
        attention_mask = torch.cat([inputs["attention_mask"], torch.ones(inputs["attention_mask"].shape[0], 1).to(inputs["input_ids"].device.type)], dim=-1)

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
) -> Tuple[Dict[str, torch.Tensor], torch.FloatTensor]:
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

    # Concat `input_ids`
    if torch.all(acceptance_mask):
        input_ids = torch.cat([inputs["input_ids"], next_token], dim=-1)
        attention_mask = torch.cat([inputs["attention_mask"], torch.ones(inputs["attention_mask"].shape[0], 1).to(inputs["input_ids"].device.type)], dim=-1)
    else:
        new_input_ids = []
        new_attention_mask = []

        for batch_idx in range(next_tokens.shape[0]):
            for pos_idx in range(acceptance_mask[batch_idx].shape[0]):
                if not acceptance_mask[batch_idx][pos_idx]:
                    gamma = next_tokens.shape[1] - 1
                    start_idx = inputs["input_ids"].shape[1] - gamma

                    inputs["input_ids"][batch_idx][start_idx+pos_idx] = next_tokens[batch_idx][pos_idx]

                    new_input_ids.append(inputs["input_ids"][batch_idx][:start_idx+pos_idx+1])
                    new_attention_mask.append(inputs["attention_mask"][batch_idx][:start_idx+pos_idx+1])
                    
                    break

        input_ids = pad_sequence(new_input_ids, batch_first=True, padding_value=target_tokenizer.pad_token_id)
        attention_mask = pad_sequence(new_attention_mask, batch_first=True, padding_value=0)

    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask

    return inputs


if __name__ == "__main__":
    # Settings
    target_model_path = "../models/HuggingFaceTB--SmolLM2-1.7B-Instruct/"
    draft_model_path = "../models/HuggingFaceTB--SmolLM2-135M-Instruct/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Tokenizer
    draft_tokenizer = GPT2TokenizerFast.from_pretrained(draft_model_path)
    target_tokenizer = GPT2TokenizerFast.from_pretrained(target_model_path)

    # Load Model
    draft_model = LlamaForCausalLM.from_pretrained(draft_model_path, torch_dtype=torch.bfloat16).to(device)
    target_model = LlamaForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.bfloat16).to(device)

    # Tokenize
    messages = [
        [
            {
                "role": "user",
                "content": "What is the capital of Taiwan. And why?",
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


    # Draft model
    target_inputs, draft_probs = drafter_speculative_decode(
        draft_model=draft_model,
        draft_tokenizer=draft_tokenizer,
        inputs=inputs,
        gamma=10,
    )

    print(target_inputs["input_ids"])
    print("".join(draft_tokenizer.batch_decode(target_inputs["input_ids"][0])))

    # Target model
    outputs = target_speculative_decode(
        target_model=target_model,
        target_tokenizer=target_tokenizer,
        inputs=target_inputs,
        draft_probs=draft_probs,
    )

    print(outputs["input_ids"])
    print("".join(target_tokenizer.batch_decode(outputs["input_ids"][0])))
