from typing import Tuple

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_repetition_penalty(
    logits: torch.FloatTensor,
    prefix_token_ids: torch.LongTensor,
    repetition_penalty: float = 1.0,
) -> torch.FloatTensor:
    batch_size, gamma, vocab_size = logits.shape
    seq_length = prefix_token_ids.shape[1]

    for batch_idx in range(batch_size):
        for gamma_idx in range(gamma):
            current_prefix = prefix_token_ids[batch_idx, :seq_length - gamma + gamma_idx + 1]

            unique_token_ids = set(current_prefix.tolist())

            for token_id in unique_token_ids:
                if logits[batch_idx, gamma_idx, token_id] > 0:
                    logits[batch_idx, gamma_idx, token_id] /= repetition_penalty
                else:
                    logits[batch_idx, gamma_idx, token_id] *= repetition_penalty

    return logits

def top_k_filtering(logits: torch.FloatTensor, top_k: int) -> torch.FloatTensor:
    if top_k > 0:
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, :, -1].unsqueeze(dim=-1)
        logits = torch.where(logits < min_values, torch.full_like(logits, -float("Inf")), logits)

    return logits


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Find the position of accumulation probs > top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    
    # Get at least one element
    sorted_indices_to_remove[:, :, 1:] = sorted_indices_to_remove[:, :, :-1].clone()
    sorted_indices_to_remove[:, :, 0] = False
    
    # Create the mask that have the same shape of logits
    indices_to_remove = sorted_indices_to_remove.scatter(2, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    
    return logits


def sample_next_token(
    logits: torch.FloatTensor,
    prefix_token_ids: torch.LongTensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    eps: float = 1e-7,
    probs_num: int = 1,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    curr_logits = logits[:, -probs_num:, :]

    # Apply repetition penalty
    if repetition_penalty != 1.0:
        curr_logits = apply_repetition_penalty(
            logits=curr_logits,
            prefix_token_ids=prefix_token_ids,
            repetition_penalty=repetition_penalty,
        )

    # Apply temperature
    curr_logits = curr_logits / (temperature + eps)

    # Apply `top_k`
    curr_logits = top_k_filtering(logits=curr_logits, top_k=top_k)

    # Apply `top_p`
    curr_logits = top_p_filtering(logits=curr_logits, top_p=top_p)

    # Convert logits into probs
    probs = torch.softmax(curr_logits, dim=-1)

    # Sampling
    seq_tokens = []
    for seq_idx in range(probs.shape[1]):
        seq_token = torch.multinomial(probs[:, seq_idx, :], num_samples=1)
        seq_tokens.append(seq_token)

    seq_token_ids = torch.cat(seq_tokens, dim=1)

    return seq_token_ids, probs


if __name__ == "__main__":
    # Settings
    pretrained_model_name_or_path = "openai-community/gpt2"
    temperature = 0.1
    top_k = 50
    top_p = 0.9
    repetition_penalty = 1.2

    # Model & Tokenizer
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Test data
    sentences = [
        "Today is a nice day",
        "How are you?",
    ]

    inputs = tokenizer(
        sentences,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to("cuda:0")

    print("=== My Sampling ===")
    for idx in range(10):
        outputs = model(**inputs)

        next_tokens, probs = sample_next_token(
            outputs.logits,
            prefix_token_ids=inputs["input_ids"],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        input_ids = torch.cat([inputs["input_ids"], next_tokens[:, -1:]], dim=-1)
        attention_mask = torch.cat([inputs["attention_mask"], torch.ones(inputs["attention_mask"].shape[0], 1).to(inputs["input_ids"].device.type)], dim=-1)

        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask


    for sent in tokenizer.batch_decode(inputs.input_ids):
        print(sent)

    print("\n=== HuggingFace ===")


    # Test data
    sentences = [
        "Today is a nice day",
        "How are you?",
    ]

    inputs = tokenizer(
        sentences,
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to("cuda:0")

    outputs = model.generate(
        **inputs,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    for sent in tokenizer.batch_decode(outputs):
        print(sent)