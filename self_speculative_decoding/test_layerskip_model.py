import time

import torch
from transformers import AutoTokenizer
from layerskip_modeling.modeling_layerskip_llama import LayerSkipLlamaForCausalLM


if __name__ == "__main__":
    pretrained_model_name_or_path = "../models/HuggingFaceTB--SmolLM2-1.7B-Instruct/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    model = LayerSkipLlamaForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16).to(device)

    skip_layer_ids = {
        "attn": [
            2,
            15,
            18,
        ],
        "mlp": [
            2,
            15,
            18,
        ]
    }

    model.set_skip_layer_ids(skip_layer_ids=skip_layer_ids)

    messages = [
        [
            {
                "role": "user",
                "content": "What is the capital of Taiwan. And why?",
            },
        ],
    ]

    # Tokenize
    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    ).to(device)

    prompt_token_num = inputs["input_ids"].shape[-1]

    # Original Model
    model.set_draft_mode(False)

    start_time = time.time()

    outputs = model.generate(**inputs, max_new_tokens=512)
    total_token_num = outputs.shape[-1]
    completion_token_num = total_token_num - prompt_token_num
    cost_time = time.time() - start_time

    token_per_second = completion_token_num / cost_time
    response = tokenizer.batch_decode(outputs)[0]

    print(f"{'='*15} Original Model {'='*15}")
    print(response)
    print()
    print(f"Completion Token Number: {completion_token_num}")
    print(f"Cost Time: {cost_time}, Speed: {token_per_second} token/sec\n")


    # LayerSkip Model
    model.set_draft_mode(True)

    start_time = time.time()

    outputs = model.generate(**inputs, max_new_tokens=512)
    total_token_num = outputs.shape[-1]
    completion_token_num = total_token_num - prompt_token_num
    cost_time = time.time() - start_time

    token_per_second = completion_token_num / cost_time
    response = tokenizer.batch_decode(outputs)[0]

    print(f"{'='*15} LayerSkip Model {'='*15}")
    print(response)
    print()
    print(f"Completion Token Number: {completion_token_num}")
    print(f"Cost Time: {cost_time}, Speed: {token_per_second} token/sec\n")
