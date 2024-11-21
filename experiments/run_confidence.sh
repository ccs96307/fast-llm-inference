#!/bin/bash


python3 speculative_decoding/run_speculative_decoding_target_confidence.py \
    --target_model_path ./models/HuggingFaceTB--SmolLM2-1.7B-Instruct \
    --draft_model_path ./models/HuggingFaceTB--SmolLM2-135M-Instruct \
    --device cuda:0 \
    --question 'What is the capital of Taiwan. And why?' \
    --gamma 5 \
    --test_token_num 100 