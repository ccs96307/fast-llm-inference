from typing import Dict, List, Optional, Tuple

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import time

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kangaroo_modeling.modeling_kangaroo_llama3 import KangarooLlamaForCausalLM
from sampling.sampling import sample_next_token


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


def run_test() -> None:
    # Device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model path 
    pretrained_model_name_or_path = "../models/meta-llama--Meta-Llama-3.1-8B-Instruct"
    adapter_dir = "checkpoints/epoch_5/"

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    model = KangarooLlamaForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16).to(device)
    model.set_skip_layer(shallow_layer_num=10)
    if adapter_dir:
        model.load_adapter(adapter_dir)

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
        model(**inputs_dummy)
    torch.cuda.synchronize()

    # Record
    raw_inputs = copy.deepcopy(inputs)
    raw_token_num = raw_inputs["input_ids"].shape[1]
    start_time = time.time()

    max_new_tokens = 100
    outputs = model.kangaroo_generate(
        **inputs,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
    )

    generate_token_num = outputs["input_ids"].shape[1] - raw_token_num
    spent_time = time.time() - start_time

    print(f"Generate token number: {generate_token_num}")
    print(f"Generate speed: {generate_token_num / spent_time} tokens/sec")
    print(f"Speculative Decoding Spent Time: {spent_time} seconds.")

    model.set_target_mode()

    # Normal Target Model Speed
    raw_inputs = copy.deepcopy(inputs)
    start_time = time.time()
    target_inputs, draft_probs = drafter_speculative_decode(
        draft_model=model,
        draft_tokenizer=tokenizer,
        inputs=raw_inputs,
        gamma=max_new_tokens,
    )

    spent_time = time.time() - start_time

    print(f"Generate token number: {max_new_tokens}")
    print(f"Generate speed: {max_new_tokens / spent_time} tokens/sec")
    print(f"Normal Target Model Decoding Spent Time: {spent_time} seconds.\n")



if __name__ == "__main__":
    run_test()
