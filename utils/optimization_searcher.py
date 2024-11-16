from typing import Dict, List, Callable, Optional

import gc
import copy
import json
import time

import optuna
import torch
from tqdm import tqdm

from utils.utils import AdaptiveDraftExitAduster


class LayerSkipStrategySearcher:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device,
        drafter_speculative_decode: Callable,
        target_speculative_decode: Callable,
        adjuster: Optional[AdaptiveDraftExitAduster] = None,
    ) -> None:
        self.model = model
        if not hasattr(self.model, "set_draft_mode"):
            raise TypeError(f"{type(self.model)} is not a LayerSkip-liked model.")
        
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize skip layers
        self.reset_skip_layers()
        
        # Sampling methods
        self.drafter_speculative_decode = drafter_speculative_decode
        self.target_speculative_decode = target_speculative_decode
        
        # Threshold adjuster
        self.adjuster = adjuster
        
        # Cache for processed samples
        self._processed_samples = None

    def reset_skip_layers(self):
        """Reset skip layers to initial state"""
        self.skip_layer_ids = {
            "attn": [],
            "mlp": [],
        }
    
    @property
    def processed_samples(self):
        """Lazy loading of processed samples with proper cleanup"""
        if self._processed_samples is None:
            raise ValueError("Samples not initialized. Call prepare_samples first.")
        return self._processed_samples
    
    def prepare_samples(self, samples: List[List[Dict[str, str]]]):
        """Process and prepare samples with proper cleanup"""
        # Clear any existing processed samples
        if self._processed_samples is not None:
            del self._processed_samples
            self._processed_samples = None
            torch.cuda.empty_cache()
            gc.collect()
        
        # Process new samples
        self._processed_samples = [
            self.tokenizer(
                self.tokenizer.apply_chat_template(messages, tokenize=False),
                return_tensors="pt",
            ).to(self.device)
            for messages in samples
        ]

    def cleanup(self):
        """Clean up resources"""
        del self._processed_samples
        self._processed_samples = None
        torch.cuda.empty_cache()
        gc.collect()

    def optimize_acceptance_rate(
        self,
        samples: List[List[Dict[str, str]]],
        n_trials: int = 50,
        num_hidden_layers: int = 1,
    ) -> Dict:
        try:
            # Prepare samples
            self.prepare_samples(samples)
            
            # Define search space
            num_hidden_layers = getattr(self.model.config, "num_hidden_layers", num_hidden_layers)
            print(f"Total layers we can skip: {num_hidden_layers}")
            
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self.objective_acceptance_rate(
                    trial=trial,
                    num_hidden_layers=num_hidden_layers,
                ),
                n_trials=n_trials,
                callbacks=[
                    lambda study, trial: print(f"Trial {trial.number}: {trial.value}"),
                    lambda study, trial: gc.collect(),  # Force garbage collection after each trial
                    lambda study, trial: torch.cuda.empty_cache()  # Clear CUDA cache after each trial
                ]
            )
            
            return {
                "best_params": study.best_params,
                "best_value": study.best_value
            }
        finally:
            self.cleanup()

    def optimize_speculative_speed(
        self,
        samples: List[List[Dict[str, str]]],
        n_trials: int = 50,
        num_hidden_layers: int = 1,
    ) -> Dict:
        try:
            # Prepare samples
            self.prepare_samples(samples)
            
            # Define search space
            num_hidden_layers = getattr(self.model.config, "num_hidden_layers", num_hidden_layers)
            print(f"Total layers we can skip: {num_hidden_layers}")
            
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self.objective_speculative_speed(
                    trial=trial,
                    num_hidden_layers=num_hidden_layers,
                ),
                n_trials=n_trials,
                callbacks=[
                    lambda study, trial: print(f"Trial {trial.number}: {trial.value}"),
                    lambda study, trial: gc.collect(),
                    lambda study, trial: torch.cuda.empty_cache()
                ]
            )

            # Save best skip_layer_ids
            best_skip_layers = self._get_skip_layers(study.best_trial, num_hidden_layers)
            with open("skip_layer_ids.json", "w") as f:
                json.dump(best_skip_layers, f)
            
            return {
                "best_params": study.best_params,
                "best_value": study.best_value
            }
        finally:
            self.cleanup()

    def _run_inference(self, inputs, gamma=5, max_new_tokens=100):
        """Common inference logic for both objectives"""
        is_end = False

        trial_inputs = copy.deepcopy(inputs)
        raw_token_num = trial_inputs["input_ids"].shape[1]
        total_draft_tokens = 0
        total_accept_tokens = 0
        generated_tokens = 0
        
        while not is_end:
            # Draft model inference
            with torch.no_grad():  # Ensure no gradients are tracked
                target_inputs, draft_probs, real_generated_tokens = self.drafter_speculative_decode(
                    draft_model=self.model,
                    draft_tokenizer=self.tokenizer,
                    inputs=trial_inputs,
                    gamma=gamma,
                    confidence_threshold_adjuster=self.adjuster,
                )
            
            total_draft_tokens += real_generated_tokens if real_generated_tokens is not None else gamma
            
            # Target model inference
            with torch.no_grad():
                outputs, is_end, accept_tokens = self.target_speculative_decode(
                    target_model=self.model,
                    target_tokenizer=self.tokenizer,
                    inputs=target_inputs,
                    draft_probs=draft_probs,
                )
            
            if self.adjuster:
                self.adjuster.update(
                    num_matched_tokens=accept_tokens,
                    num_drafted_tokens=real_generated_tokens if real_generated_tokens is not None else gamma,
                )
            
            total_accept_tokens += accept_tokens
            trial_inputs = outputs
            
            generated_tokens = trial_inputs["input_ids"].shape[1] - raw_token_num
            if generated_tokens >= max_new_tokens:
                break
            
        # Free memory
        del target_inputs, draft_probs, trial_inputs
        torch.cuda.empty_cache()
        
        return total_accept_tokens, total_draft_tokens, generated_tokens

    def objective_speculative_speed(self, trial, num_hidden_layers: int):
        try:
            skip_layers = self._get_skip_layers(trial, num_hidden_layers)
            if not skip_layers:
                raise optuna.TrialPruned()
            
            self.model.set_skip_layer_ids(skip_layer_ids=skip_layers)
            
            if self.adjuster:
                self.adjuster.reset()
            
            start_time = time.time()
            total_generated_tokens = 0
            
            for inputs in tqdm(self.processed_samples):
                _, _, generated_tokens = self._run_inference(inputs)
                total_generated_tokens += generated_tokens
            
            token_per_second = total_generated_tokens / (time.time() - start_time)
            print(f"attn_skip: {skip_layers['attn']}, mlp_skip: {skip_layers['mlp']}, tokens/sec: {token_per_second}")
            
            return token_per_second
        
        except Exception as e:
            print(f"Trial failed with error: {str(e)}")
            raise optuna.TrialPruned()

    def objective_acceptance_rate(self, trial, num_hidden_layers: int):
        try:
            skip_layers = self._get_skip_layers(trial, num_hidden_layers)
            if not skip_layers:
                raise optuna.TrialPruned()
            
            self.model.set_skip_layer_ids(skip_layer_ids=skip_layers)
            
            total_accept_tokens_group = 0
            total_draft_tokens_group = 0
            
            for inputs in tqdm(self.processed_samples):
                accept_tokens, draft_tokens, _ = self._run_inference(inputs)
                total_accept_tokens_group += accept_tokens
                total_draft_tokens_group += draft_tokens
            
            accept_rate = total_accept_tokens_group / total_draft_tokens_group
            print(f"attn_skip: {skip_layers['attn']}, mlp_skip: {skip_layers['mlp']}, Accept Rate: {accept_rate}")
            
            return accept_rate
        
        except Exception as e:
            print(f"Trial failed with error: {str(e)}")
            raise optuna.TrialPruned()

    def _get_skip_layers(self, trial, num_hidden_layers: int) -> Dict[str, List[int]]:
        """Helper method to get skip layers configuration from trial"""
        skip_attn_layers = [i for i in range(num_hidden_layers) if trial.suggest_int(f"skip_attn_layer_{i}", 0, 1) == 1]
        skip_mlp_layers = [i for i in range(num_hidden_layers) if trial.suggest_int(f"skip_mlp_layer_{i}", 0, 1) == 1]
        
        if not skip_attn_layers and not skip_mlp_layers:
            return None
            
        return {
            "attn": skip_attn_layers,
            "mlp": skip_mlp_layers,
        }
