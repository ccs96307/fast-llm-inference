from typing import Any, List, Optional, Tuple, Union

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dataclasses import dataclass
import json

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import Cache, DynamicCache, LlamaSdpaAttention
from transformers.modeling_outputs import CausalLMOutputWithPast

from sampling.sampling import sample_next_token
from utils.utils import calculate_continuous_acceptance


@dataclass
class KangarooModelMode:
    draft_only_mode: str = "draft_only"
    target_only_mode: str = "target_only"
    train_mode: str = "train"


class KangarooLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.draft_mode_adapter_layer = LlamaSdpaAttention(config=config, layer_idx=config.num_hidden_layers)
        self.shallow_layers = None
        self.mode = KangarooModelMode.target_only_mode
        self.confidence_threshold = 0.5
        self.accept_rate = 0
        self.total_accept_tokens = 0
        self.total_draft_generated_token = 0
        self.temperature = 1.0
        self.alpha = 0
        self.shallow_layer_num = 10
    
    def set_skip_layer(self, shallow_layer_num: int) -> None:
        self.shallow_layer_num = shallow_layer_num
        self.shallow_layers = self.model.layers[:shallow_layer_num]
        self.remaining_layers = self.model.layers[shallow_layer_num:]

    def set_draft_mode(self) -> None:
        self.mode = KangarooModelMode.draft_only_mode

    def set_target_mode(self) -> None:
        self.mode = KangarooModelMode.target_only_mode

    def set_train_mode(self) -> None:
        self.mode = KangarooModelMode.train_mode

    def save_adapter(
        self,
        save_dir: str,
        train_loss_history: List[float],
        eval_loss_history: List[float],
    ) -> None:
        """
        Save the parameters of the draft_mode_adapter_layer, loss_history, and shallow_layer_num to the specified directory.

        Args:
            save_dir (str): Directory to save the adapter parameters.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the adapter's state dictionary
        adapter_path = os.path.join(save_dir, "draft_adapter.pt")
        torch.save(self.draft_mode_adapter_layer.state_dict(), adapter_path)
        print(f"Draft adapter saved at {adapter_path}")

        # Save additional information (loss_history and shallow_layer_num)
        metadata = {
            "train_loss_history": train_loss_history,
            "eval_loss_history": eval_loss_history,
            "shallow_layer_num": self.shallow_layer_num
        }
        metadata_path = os.path.join(save_dir, "adapter_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        print(f"Adapter metadata saved at {metadata_path}")

    def load_adapter(self, load_dir: str) -> None:
        """
        Load the parameters of the draft_mode_adapter_layer, loss_history, and shallow_layer_num from the specified directory.

        Args:
            load_dir (str): Directory to load the adapter parameters from.

        Raises:
            FileNotFoundError: If the adapter file does not exist in the specified directory.
        """
        adapter_path = os.path.join(load_dir, "draft_adapter.pt")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Draft adapter not found at {adapter_path}")
        
        # Load the adapter's state dictionary
        state_dict = torch.load(adapter_path, map_location=self.device)
        self.draft_mode_adapter_layer.load_state_dict(state_dict=state_dict)
        print(f"Draft adapter loaded from {adapter_path}")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.mode == KangarooModelMode.target_only_mode:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
                **loss_kwargs,
            )
        elif self.mode == KangarooModelMode.draft_only_mode:
            if self.shallow_layers is None:
                raise AttributeError(f"You do not set the `shallow_layers`!")

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # Model
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if self.model.gradient_checkpointing and self.model.training and use_cache:
                use_cache = False

            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)

            # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = False
            if use_cache and not isinstance(past_key_values, Cache):
                return_legacy_cache = True
                if past_key_values is None:
                    past_key_values = DynamicCache()
                else:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            causal_mask = self.model._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )
            hidden_states = inputs_embeds

            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

            # decoder layers
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            next_decoder_cache = None

            for decoder_layer in self.shallow_layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            # Adapter
            residual = hidden_states
            hidden_states = self.model.norm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            next_cache = next_decoder_cache if use_cache else None
            if return_legacy_cache:
                next_cache = next_cache.to_legacy_cache()

            hidden_states, self_attn_weights, past_key_values = self.draft_mode_adapter_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = residual + hidden_states
            hidden_states = self.model.norm(hidden_states)

            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [torch.nn.functional.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
                logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

            loss = None
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=hidden_states,
                attentions=self_attn_weights,
            )
        elif self.mode == KangarooModelMode.train_mode:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if self.model.gradient_checkpointing and self.model.training and use_cache:
                use_cache = False

            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)

            # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = False
            if use_cache and not isinstance(past_key_values, Cache):
                return_legacy_cache = True
                if past_key_values is None:
                    past_key_values = DynamicCache()
                else:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            causal_mask = self.model._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )
            hidden_states = inputs_embeds

            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

            # decoder layers
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            next_decoder_cache = None

            for decoder_layer in self.shallow_layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            # Cache hidden states
            remaining_hidden_states = hidden_states

            # Attention adapter
            residual = hidden_states
            hidden_states = self.model.norm(hidden_states)

            # Add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            next_cache = next_decoder_cache if use_cache else None
            if return_legacy_cache:
                next_cache = next_cache.to_legacy_cache()

            hidden_states, self_attn_weights, past_key_values = self.draft_mode_adapter_layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            hidden_states = residual + hidden_states
            hidden_states = self.model.norm(hidden_states)

            # Remaining decoder layers
            for decoder_layer in self.remaining_layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                layer_outputs = decoder_layer(
                    remaining_hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

                remaining_hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            remaining_hidden_states = self.model.norm(remaining_hidden_states)

            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            draft_logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]) / self.temperature
            target_logits = self.lm_head(remaining_hidden_states[:, -num_logits_to_keep:, :]) / self.temperature

            # Compute the log probabilities for both models
            draft_log_probs = torch.nn.functional.log_softmax(draft_logits, dim=-1)
            target_log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)
            target_probs = torch.nn.functional.softmax(target_logits, dim=-1)

            # Cross-entropy loss between target and draft model predictions
            kl_loss = torch.nn.functional.kl_div(draft_log_probs, target_probs, reduction="batchmean") * (self.temperature ** 2)
            cross_entropy_loss = -(target_probs * draft_log_probs).sum(dim=-1).mean()
            loss = self.alpha * cross_entropy_loss + (1 - self.alpha) * kl_loss

            return CausalLMOutputWithPast(
                loss=loss,
                logits=target_logits,
                past_key_values=past_key_values,
                hidden_states=hidden_states,
                attentions=self_attn_weights,
            )

    def kangaroo_generate(
        self,
        eos_token_id: int,
        pad_token_id: int,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> torch.LongTensor:
        if self.shallow_layers is None:
            raise AttributeError(f"You do not set the `shallow_layers`!")

        confidence_score = 1.0
        total_generate_tokens = 0
        
        while total_generate_tokens < max_new_tokens:
            draft_generate_tokens = 0
            draft_probs = []

            while confidence_score >= self.confidence_threshold:
                output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
                output_hidden_states = (
                    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                )
                return_dict = return_dict if return_dict is not None else self.config.use_return_dict

                if (input_ids is None) ^ (inputs_embeds is not None):
                    raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

                if self.model.gradient_checkpointing and self.model.training and use_cache:
                    use_cache = False

                if inputs_embeds is None:
                    inputs_embeds = self.model.embed_tokens(input_ids)

                # kept for BC (non `Cache` `past_key_values` inputs)
                return_legacy_cache = False
                if use_cache and not isinstance(past_key_values, Cache):
                    return_legacy_cache = True
                    if past_key_values is None:
                        past_key_values = DynamicCache()
                    else:
                        past_key_values = DynamicCache.from_legacy_cache(past_key_values)

                if cache_position is None:
                    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                    cache_position = torch.arange(
                        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                    )
                if position_ids is None:
                    position_ids = cache_position.unsqueeze(0)

                causal_mask = self.model._update_causal_mask(
                    attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
                )
                hidden_states = inputs_embeds

                # create position embeddings to be shared across the decoder layers
                position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

                # decoder layers
                all_hidden_states = () if output_hidden_states else None
                all_self_attns = () if output_attentions else None
                next_decoder_cache = None

                for decoder_layer in self.shallow_layers:
                    if output_hidden_states:
                        all_hidden_states += (hidden_states,)

                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )

                    hidden_states = layer_outputs[0]

                    if use_cache:
                        next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                    if output_attentions:
                        all_self_attns += (layer_outputs[1],)

                # Cache hidden states
                remaining_hidden_states = hidden_states

                # Attention adapter
                residual = hidden_states
                hidden_states = self.model.norm(hidden_states)

                # Add hidden states from the last decoder layer
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                next_cache = next_decoder_cache if use_cache else None
                if return_legacy_cache:
                    next_cache = next_cache.to_legacy_cache()

                hidden_states, self_attn_weights, past_key_values = self.draft_mode_adapter_layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

                hidden_states = residual + hidden_states
                hidden_states = self.model.norm(hidden_states)

                # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
                draft_logits = self.lm_head(hidden_states[:, -1:, :])

                # Sampling and get the probabilities
                next_tokens, probs = sample_next_token(
                    logits=draft_logits,
                    prefix_token_ids=input_ids,
                )

                draft_probs.append(probs)
                input_ids = torch.cat([input_ids, next_tokens[:, -1:]], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones(attention_mask.shape[0], 1).to(input_ids.device)], dim=-1)

                draft_generate_tokens += 1
                self.total_draft_generated_token += 1

                # Re-init
                inputs_embeds = None
                position_ids = None
                cache_position = None

                # Support bs=1
                decode_token_id = next_tokens[:, -1].item()
                if probs[:, -1, decode_token_id] < self.confidence_threshold or total_generate_tokens + draft_generate_tokens >= max_new_tokens:
                    draft_probs = torch.cat(draft_probs, dim=1)
                    break

            # Use whole model for evaluating
            for decoder_layer in self.remaining_layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                layer_outputs = decoder_layer(
                    remaining_hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

                remaining_hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            remaining_hidden_states = self.model.norm(remaining_hidden_states)
            num_logits_to_keep = draft_probs.shape[1]
            target_logits = self.lm_head(remaining_hidden_states[:, -num_logits_to_keep:, :])

            target_input_ids = input_ids[:, :-1]

            next_tokens, target_probs = sample_next_token(
                logits=target_logits,
                prefix_token_ids=target_input_ids,
                probs_num=num_logits_to_keep,
            )

            # Evaluation
            expanded_indices = input_ids[:, -draft_probs.shape[1]:].unsqueeze(-1)

            # Get each probilities
            selected_draft_probs = torch.gather(draft_probs, dim=-1, index=expanded_indices).squeeze(-1)
            selected_eval_probs = torch.gather(target_probs, dim=-1, index=expanded_indices).squeeze(-1)

            # Compare draft_prob and eval_prob, and check the reject_mask
            mask_to_reject = selected_draft_probs > selected_eval_probs

            # Calculate reject probabilty 1 - (eval_prob / draft_prob)
            rejection_probs = 1 - (selected_eval_probs / selected_draft_probs)

            # Generate random values to determined accept or reject
            random_values = torch.rand_like(rejection_probs)
            rejection_decisions = random_values < rejection_probs

            # Get the final reject masks
            rejection_masks = mask_to_reject & rejection_decisions
            acceptance_mask = torch.ones_like(selected_draft_probs, dtype=torch.bool)
            acceptance_mask[rejection_masks] = False

            # Concat `input_ids`
            if torch.all(acceptance_mask):
                total_generate_tokens += draft_generate_tokens
            else:
                new_input_ids = []
                new_attention_mask = []
                is_end = False

                for batch_idx in range(next_tokens.shape[0]):
                    gamma = next_tokens.shape[1]
                    start_idx = input_ids.shape[1] - gamma

                    for pos_idx in range(acceptance_mask[batch_idx].shape[0]):
                        total_generate_tokens += 1
                        if (acceptance_mask[batch_idx][pos_idx] and input_ids[batch_idx][start_idx+pos_idx].item() == eos_token_id) or not acceptance_mask[batch_idx][pos_idx]:
                            input_ids[batch_idx][start_idx+pos_idx] = next_tokens[batch_idx][pos_idx]

                            new_input_ids.append(input_ids[batch_idx][:start_idx+pos_idx+1])
                            new_attention_mask.append(attention_mask[batch_idx][:start_idx+pos_idx+1])
                            
                            is_end = input_ids[batch_idx][start_idx+pos_idx].item() == eos_token_id
                            break

                input_ids = pad_sequence(new_input_ids, batch_first=True, padding_value=pad_token_id)
                attention_mask = pad_sequence(new_attention_mask, batch_first=True, padding_value=0)

            self.total_accept_tokens += calculate_continuous_acceptance(acceptance_mask=acceptance_mask)
            self.accept_rate = self.total_accept_tokens / self.total_draft_generated_token
        
            if is_end:
                break

        return {"input_ids": input_ids}
