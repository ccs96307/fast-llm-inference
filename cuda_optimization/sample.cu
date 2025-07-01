#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cmath>

namespace cg = cooperative_groups;

// --- Helper Functions within the CUDA device context ---

// A simple device function to check if a token is in the prefix
__device__ inline bool in_prefix(long token_id, const long* prefix_ptr, int prefix_len) {
    for (int i = 0; i < prefix_len; ++i) {
        if (prefix_ptr[i] == token_id) {
            return true;
        }
    }
    return false;
}

// --- The Main Fused Kernel ---

__global__ void fused_sampling_kernel(
    const float* __restrict__ logits_ptr,      // Input: [batch_size, vocab_size]
    long* __restrict__ output_tokens_ptr,      // Output: [batch_size, 1]
    const long* __restrict__ prefix_tokens_ptr,// Input: [batch_size, prefix_len]
    int vocab_size,
    int prefix_len,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty
) {
    // --- Setup ---
    const int batch_idx = blockIdx.x;
    const int thread_idx = threadIdx.x;
    const int block_size = blockDim.x;

    // A cooperative group for this block
    cg::thread_block block = cg::this_thread_block();

    // Shared memory for this block to work on one sequence's logits
    extern __shared__ float shared_mem[];
    float* shared_logits = shared_mem; // size: vocab_size
    // Use part of shared memory for indices for sorting
    int* shared_indices = (int*)(shared_logits + vocab_size); // size: vocab_size

    // Pointers for the current batch item
    const float* current_logits = logits_ptr + batch_idx * vocab_size;
    const long* current_prefix = prefix_tokens_ptr + batch_idx * prefix_len;
    
    // --- Step 1: Parallel Load, Repetition Penalty & Temperature ---
    float max_logit = -FLT_MAX;
    for (int i = thread_idx; i < vocab_size; i += block_size) {
        float logit = current_logits[i];

        // Apply repetition penalty
        if (repetition_penalty != 1.0f && in_prefix(i, current_prefix, prefix_len)) {
            if (logit > 0) {
                logit /= repetition_penalty;
            } else {
                logit *= repetition_penalty;
            }
        }
        
        // Apply temperature
        logit /= temperature;

        shared_logits[i] = logit;
        shared_indices[i] = i;
        // Find max logit in parallel for stable softmax
        max_logit = fmaxf(max_logit, logit);
    }

    // --- Step 2: Parallel Reductions for Softmax and Top-K/P ---
    // Reduce to find the single max_logit for the entire block
    cg::reduce(block, &max_logit, cg::max<float>());
    cg::sync(block);

    // --- Step 3: Softmax Calculation ---
    float exp_sum = 0.0f;
    for (int i = thread_idx; i < vocab_size; i += block_size) {
        shared_logits[i] = expf(shared_logits[i] - max_logit);
        exp_sum += shared_logits[i];
    }
    cg::reduce(block, &exp_sum, cg::plus<float>());
    cg::sync(block);

    // Normalize to get probabilities
    for (int i = thread_idx; i < vocab_size; i += block_size) {
        shared_logits[i] /= exp_sum;
    }
    cg::sync(block);

    // --- Step 4: Top-K & Top-P Filtering ---
    // This is a simplified sort (block-wide bitonic sort would be more robust)
    // For simplicity, we perform a bubble sort, which is slow but easy to understand.
    // A production kernel would use a more efficient parallel sort.
    if (top_k > 0 || top_p < 1.0f) {
        for (int i = 0; i < vocab_size; ++i) {
            for (int j = thread_idx; j < vocab_size - i - 1; j += block_size) {
                if (shared_logits[j] < shared_logits[j+1]) {
                    float temp_val = shared_logits[j];
                    shared_logits[j] = shared_logits[j+1];
                    shared_logits[j+1] = temp_val;

                    int temp_idx = shared_indices[j];
                    shared_indices[j] = shared_indices[j+1];
                    shared_indices[j+1] = temp_idx;
                }
            }
            cg::sync(block);
        }
    }
    
    // Apply top-k: zero out probabilities beyond k
    if (top_k > 0) {
        for (int i = thread_idx; i < vocab_size; i += block_size) {
            if (i >= top_k) {
                shared_logits[i] = 0.0f;
            }
        }
    }
    cg::sync(block);

    // Apply top-p: find cutoff and zero out probabilities
    if (top_p < 1.0f) {
        float cumulative_p = 0.0f;
        // This part needs a scan (prefix sum), doing it serially in thread 0 for simplicity
        if (thread_idx == 0) {
            for (int i = 0; i < vocab_size; ++i) {
                cumulative_p += shared_logits[i];
                if (cumulative_p > top_p) {
                    // Zero out the rest
                    for (int j = i + 1; j < vocab_size; ++j) {
                        shared_logits[j] = 0.0f;
                    }
                    break;
                }
            }
        }
    }
    cg::sync(block);

    // --- Step 5: Re-normalize and Sample ---
    // Re-calculate sum after filtering
    float final_sum = 0.0f;
    for (int i = thread_idx; i < vocab_size; i += block_size) {
        final_sum += shared_logits[i];
    }
    cg::reduce(block, &final_sum, cg::plus<float>());
    cg::sync(block);

    // Final normalization
    for (int i = thread_idx; i < vocab_size; i += block_size) {
        shared_logits[i] /= final_sum;
    }
    cg::sync(block);

    // --- Step 6: Multinomial Sampling ---
    if (thread_idx == 0) {
        // Initialize cuRAND state
        curandState_t state;
        curand_init(blockIdx.x, 0, 0, &state);
        
        // Generate a random number
        float rand_val = curand_uniform(&state);
        
        float cumulative_prob = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            cumulative_prob += shared_logits[i];
            if (rand_val <= cumulative_prob) {
                output_tokens_ptr[batch_idx] = shared_indices[i];
                break;
            }
        }
    }
}


// --- Kernel Launcher Function ---
// This C++ function is called by the wrapper. It configures and launches the kernel.
void launch_sampling_kernel(
    torch::Tensor logits,
    torch::Tensor output_tokens,
    const torch::Tensor prefix_token_ids,
    float temperature,
    int top_k,
    float top_p,
    float repetition_penalty
) {
    const int batch_size = logits.size(0);
    const int vocab_size = logits.size(1);
    const int prefix_len = prefix_token_ids.size(1);

    // --- Kernel Configuration ---
    // Use one block per batch item.
    // Use a reasonable number of threads per block. 128 is a safe choice.
    const dim3 block_dim(128);
    const dim3 grid_dim(batch_size);

    // Shared memory size: enough for logits and indices
    const int shared_mem_size = (vocab_size * sizeof(float)) + (vocab_size * sizeof(int));

    // --- Kernel Launch ---
    fused_sampling_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
        logits.data_ptr<float>(),
        output_tokens.data_ptr<long>(),
        prefix_token_ids.data_ptr<long>(),
        vocab_size,
        prefix_len,
        temperature,
        top_k,
        top_p,
        repetition_penalty
    );

    // Check for any errors during kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
