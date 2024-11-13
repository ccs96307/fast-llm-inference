# Fast LLM Inference
# Fast LLM Inference - Optimized Task Plan

## TODO List

### November 2024
- **[x] 2024/11/08** | Complete `Speculative Decoding` following the paper [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192)
- **[ ] 2024/11/15** | Implement `Self-Speculative Decoding` as per [Draft & Verify - Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://arxiv.org/pdf/2309.08168)
  - [x] LayerSkip model architecture
  - [x] Bayesian Optimization for Layer Skip Selection
  - [ ] Optimization
- **[ ] 2024/11/22** | Develop `Kangaroo` following [Kangaroo - Lossless Self-Speculative Decoding via Double Early Exiting](https://arxiv.org/pdf/2404.18911)
  - [ ] Implement double early exits to improve speed.
  - [ ] Training Script
- **[ ] 2024/11/29** | Implement `Medusa` from [Medusa - Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/pdf/2401.10774)
  - **Goal**: Test with multi-headed architecture for versatility in decoding.

### Additional Enhancements
- **[ ] TBD** | Implement `prompt look-up decoding` per [prompt-lookup-decoding GitHub](https://github.com/apoorvumang/prompt-lookup-decoding)
- **[ ] TBD** | Implement `UAG` (Universal Assisted Generation) as per [Universal Assisted Generation Blog](https://huggingface.co/blog/universal_assisted_generation)

---

## Updates Log
### November 2024
- **2024/11/08**: `Speculative Decoding` successfully implemented. Verified improved inference time with no noticeable accuracy degradation.
- **2024/11/10**: Initial setup for `Self-Speculative Decoding` completed; data pipeline in place for testing draft-and-verify.
- **2024/11/12**: Reviewing implementation challenges for `Self-Speculative Decoding` and evaluating model compatibility for improved efficiency.

### Pending Decisions
- **Prompt look-up decoding**: Determine timeline after reviewing initial implementations.
- **UAG Integration**: Assess when to integrate after `Medusa` and `Kangaroo` are in place.
