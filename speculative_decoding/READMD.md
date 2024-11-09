# Speculative Decoding
## Introduction
The implementation is refer from: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192).

## Usage
You can use the following command to execute the simple test:
```bash
python speculative_decoding/run_speculative_decoding.py \
    --target_model_path HuggingFaceTB/SmolLM2-1.7B-Instruct \
    --draft_model_path HuggingFaceTB/SmolLM2-135M-Instruct \
    --device cuda:0 \
    --question 'What is the capital of Taiwan. And why?' \
    --gamma 5 \
    --test_token_num 100 
```