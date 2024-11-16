from typing import Dict, List, Callable

import copy

import optuna
import torch


class LayerSkipStrategySearcher:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device,
        drafter_speculative_decode: callable,
        target_speculative_decode: callable,
    ) -> None:
        self.model = model
        if not hasattr(self.model, "set_draft_mode", False):
            raise TypeError(f"{type(self.model)} is not a LayerSkip-liked model.")
        
        self.tokenizer = tokenizer
        self.device = device

        self.skip_layer_ids = {
            "attn": [],
            "mlp": [],
        }

        # Sampling methods
        self.drafter_speculative_decode = drafter_speculative_decode
        self.target_speculative_decode = target_speculative_decode

    def optimize_acceptance_rate(
        self,
        samples: List[List[Dict[str, str]]],
        n_trails: int = 50,
        num_hidden_layers: int = 1,
    ):
        # Define search space
        num_hidden_layers = getattr(self.model.config, "num_hidden_layers", num_hidden_layers)
        print(f"Total layers we can skip: {num_hidden_layers}")

        # Tokenized
        self.processed_samples = [
            self.tokenizer(
                self.tokenizer.apply_chat_template(messages, tokenize=False),
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to(self.device)
            for messages in samples
        ]
        
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trail: self.objective_acceptance_rate(
                trail=trail,
                samples=samples,
                num_hidden_layers=num_hidden_layers,
            ),
            n_trials=n_trails,
            callbacks=[lambda study, trial: print(f"Trial {trial.number}: {trial.value}")],
        )
        print("The best params:", study.best_params)
        print("The best accept_rate:", study.best_value)

    def optimize_speculative_speed(self):
        pass

    def objective_acceptance_rate(self, trial, samples: List[List[Dict[str, str]]], num_hidden_layers: int):
        # Determine skip or not for `attn` and `mlp`
        skip_attn_layers = [i for i in range(num_hidden_layers) if trial.suggest_int(f"skip_attn_layer_{i}", 0, 1) == 1]
        skip_mlp_layers = [i for i in range(num_hidden_layers) if trial.suggest_int(f"skip_mlp_layer_{i}", 0, 1) == 1]

        # Disable set to 0 both
        if len(skip_attn_layers) == 0 and len(skip_mlp_layers) == 0:
            raise optuna.TrialPruned()

        skip_layer_ids = {
            "attn": skip_attn_layers,
            "mlp": skip_mlp_layers,
        }

        # Set the skip strategy
        self.model.set_skip_layer_ids(skip_layer_ids=skip_layer_ids)

        # Trails
        total_accept_tokens_group = 0
        total_draft_tokens_group = 0

        for inputs in self.processed_samples:
            is_end = False

            # Record
            raw_inputs = copy.deepcopy(inputs)
            raw_token_num = raw_inputs["input_ids"].shape[1]

            total_draft_tokens = 0
            total_accept_tokens = 0
            gamma = 5
            max_new_tokens = 100

            while not is_end:
                # Draft model
                target_inputs, draft_probs, _ = self.drafter_speculative_decode(
                    draft_model=self.model,
                    draft_tokenizer=self.tokenizer,
                    inputs=inputs,
                    gamma=gamma,
                )

                total_draft_tokens += gamma

                # Target model
                outputs, is_end, accept_tokens = self.target_speculative_decode(
                    target_model=self.model,
                    target_tokenizer=self.tokenizer,
                    inputs=target_inputs,
                    draft_probs=draft_probs,
                )

                total_accept_tokens += accept_tokens
                inputs = outputs

                if inputs["input_ids"].shape[1] - raw_token_num >= max_new_tokens:
                    break

            # Compute acceptance rate
            total_accept_tokens_group += total_accept_tokens
            total_draft_tokens_group += total_draft_tokens
            accept_rate = total_accept_tokens_group / total_draft_tokens_group
            print(f"attn_skip: {skip_attn_layers}, mlp_skip: {skip_mlp_layers}, Accept Rate: {accept_rate}")

        # Assume we want to maximize `accept_rate`
        return accept_rate