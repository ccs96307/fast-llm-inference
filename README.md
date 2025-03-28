# Fast LLM Inference - Optimized Task Plan
I hope to implement some acceleration technologies for Large Language Models (LLMs) because I enjoy doing this myself and love the challenge of bringing research papers into real-world applications.

If there are any technologies you'd like to develop or discuss, feel free to reach out. Thanks!

I'm excited to dive deeper into AI research!  

---

## Updates Log
### 2024
- **2024/12/16**: Add the `Medusa-1 Training Script v2`
- **2024/12/15**: Add the `Medusa-1 Training Script`
- **2024/12/12**: Update the KV Cache support for Speculative Decoding
- **2024/12/04**: Add the `Kangaroo Training Script v2`
- **2024/11/26**: Add the `Kangaroo Training Script`
- **2024/11/22**: Update the `Target Model Keep Generation Mechanism` experiment
- **2024/11/18**: Update the `Self-Speculative Decoding` experiment results of `google--gemma-2-9b-it`.
- **2024/11/12**: Reviewing implementation challenges for `Self-Speculative Decoding` and evaluating model compatibility for improved efficiency.
- **2024/11/10**: Initial setup for `Self-Speculative Decoding` completed; data pipeline in place for testing draft-and-verify.
- **2024/11/08**: `Speculative Decoding` successfully implemented. Verified improved inference time with no noticeable accuracy degradation.

### Pending Decisions
- **Batched Speculative Decoding**: 
- **Prompt lookup decoding**: Determine timeline after reviewing initial implementations.
- **UAG Integration**: Assess when to integrate after `Medusa` and `Kangaroo` are in place.

---

## TODO List
### November 2024
- [x] **2024/11/08** | Complete `Speculative Decoding` following the paper [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192)
- [x] **2024/11/15** | Implement `Self-Speculative Decoding` as per [Draft & Verify - Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://arxiv.org/pdf/2309.08168)
  - [x] LayerSkip model architecture
  - [x] Bayesian Optimization for Layer Skip Selection (AR)
  - [x] Adaption Draft-Exiting Mechanism
  - [x] Optimization
  - [x] Bayesian Optimization for Layer Skip Selection (Speed) 
  - [x] `gemma-2-9b-it` experiment
- [x] **2024/11/22** | Develop `Kangaroo` following [Kangaroo - Lossless Self-Speculative Decoding via Double Early Exiting](https://arxiv.org/pdf/2404.18911)
  - [x] Kangaroo model
  - [x] Training Script
  - [x] Implement double early exits to improve speed.
- [ ] **2024/11/29** | Implement `Medusa` from [Medusa - Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/pdf/2401.10774)
  - [X] Medusa model
  - [X] Training Script (Medusa-1)
  - [ ] Testing
- [ ] **2025/03** | Implement `Hydra` from [Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding](https://arxiv.org/abs/2402.05109)
- [ ] **2025/03** | Implement `Lookahead Decoding` from [Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057)
- [ ] **2025/04** | Implement `Eagle` from [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- [ ] **2025/04** | Implement `Eagle-2` from [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858)
- [ ] **2025/04** | Implement `Eagle-3` from [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840)


### Additional Enhancements
- [ ] **TBD** | Implement `Batched Speculative Decoding` from [The Synergy of Speculative Decoding and Batching in Serving Large Language Models](https://arxiv.org/pdf/2310.18813)
- [ ] **TBD** | Implement `prompt lookup decoding` from [prompt-lookup-decoding GitHub](https://github.com/apoorvumang/prompt-lookup-decoding)
- [ ] **TBD** | Implement `UAG` (Universal Assisted Generation) from [Universal Assisted Generation Blog](https://huggingface.co/blog/universal_assisted_generation)

