from typing import Any, List, Optional, Tuple, Union

import os
from dataclasses import dataclass

import torch
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import Cache, DynamicCache, LlamaSdpaAttention
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class KangarooModelMode:
    draft_only_mode: str = "draft_only"
    target_only_mode: str = "target_only"
    accelerate_mode: str = "accelerate"
    train_mode: str = "train"


class KangarooLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.draft_mode_adapter_layer = LlamaSdpaAttention(config=config, layer_idx=config.num_hidden_layers)
        self.shallow_layers = None
        self.mode = KangarooModelMode.target_only_mode
    
    def set_skip_layer(self, shallow_layer_num: int) -> None:
        self.shallow_layers = self.model.layers[:shallow_layer_num]
        self.remaining_layers = self.model.layers[shallow_layer_num:]

    def set_draft_mode(self) -> None:
        self.mode = KangarooModelMode.draft_only_mode

    def set_target_mode(self) -> None:
        self.mode = KangarooModelMode.target_only_mode

    def set_train_mode(self) -> None:
        self.mode = KangarooModelMode.train_mode

    def set_acceleration_mode(self) -> None:
        self.mode = KangarooModelMode.accelerate_mode

    def save_adapter(self, save_directory: str) -> None:
        """
        Save the parameters of the draft_mode_adapter_layer to the specified directory.

        Args:
            save_directory (str): Directory to save the adapter parameters.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save the adapter's state dictionary
        adapter_path = os.path.join(save_directory, "draft_adapter.pt")
        torch.save(self.draft_mode_adapter_layer.state_dict(), adapter_path)
        print(f"Draft adapter saved at {adapter_path}")

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
            return super.forward(
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
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            ### === MODEL ===
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

            print(attention_mask.shape)
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

            ### Adapter
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
            draft_logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
            target_logits = self.lm_head(remaining_hidden_states[:, -num_logits_to_keep:, :])

            # Compute the log probabilities for both models
            draft_log_probs = torch.nn.functional.log_softmax(draft_logits, dim=-1)
            target_probs = torch.nn.functional.softmax(target_logits, dim=-1)

            # Cross-entropy loss between target and draft model predictions
            loss = -(target_probs * draft_log_probs).sum(dim=-1).mean()

            return CausalLMOutputWithPast(
                loss=loss,
                logits=target_logits,
                past_key_values=past_key_values,
                hidden_states=hidden_states,
                attentions=self_attn_weights,
            )
