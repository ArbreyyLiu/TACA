# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from typing import Optional
from diffusers.models.attention_processor import Attention

try:
    from torch.nn.attention.flex_attention import flex_attention
    HAS_FLEX_ATTENTION = True
except ImportError:
    HAS_FLEX_ATTENTION = False


class RegionalDreamRendererFluxAttnProcessor2_0:
    """
    Regional Dream Renderer Flux Attention Processor with TACA temperature control integration.
    
    This processor enhances the standard Flux attention mechanism with TACA's temperature control
    to improve text-image alignment during regional control operations.
    
    Args:
        taca_scale (float, optional): Temperature scale factor for TACA attention. Defaults to 1.2.
        taca_encoder_size (int, optional): Encoder size threshold for TACA scaling. Defaults to 512.
        enable_taca_temp_control (bool, optional): Whether to enable TACA temperature control. Defaults to True.
    """
    
    def __init__(
        self,
        taca_scale: float = 1.2,
        taca_encoder_size: int = 512,
        enable_taca_temp_control: bool = True,
    ):
        self.regional_mask = None
        self.taca_scale = taca_scale
        self.taca_encoder_size = taca_encoder_size
        self.enable_taca_temp_control = enable_taca_temp_control
    
    def _create_taca_score_mod(self):
        """Create the score modification function for TACA temperature control."""
        def score_mod(score, batch, head, token_q, token_kv):
            # Apply TACA temperature scaling for image-to-text attention
            # Scale region [:, :, encoder_size:, :encoder_size] - image tokens attending to text tokens
            condition = (token_q >= self.taca_encoder_size) & (token_kv < self.taca_encoder_size)
            score = torch.where(condition, score * self.taca_scale, score)
            return score
        return score_mod
    
    def _apply_manual_taca_scaling(self, attention_scores, query_length, key_length):
        """Apply TACA temperature scaling manually when flex_attention is not available."""
        if not self.enable_taca_temp_control:
            return attention_scores
        
        # Create attention mask for TACA scaling
        # Image tokens (>= encoder_size) attending to text tokens (< encoder_size)
        device = attention_scores.device
        dtype = attention_scores.dtype
        
        # Determine text and image token boundaries
        text_tokens = min(self.taca_encoder_size, key_length)
        
        if query_length > self.taca_encoder_size:
            # Create scaling mask for image-to-text attention
            scale_mask = torch.ones_like(attention_scores)
            
            # Scale the region where image tokens attend to text tokens
            image_start = self.taca_encoder_size
            if image_start < query_length:
                scale_mask[..., image_start:, :text_tokens] = self.taca_scale
                attention_scores = attention_scores * scale_mask
        
        return attention_scores
    
    def FluxAttnProcessor2_0_call(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        base_ratio: Optional[float] = None,
    ) -> torch.FloatTensor:
        """
        Enhanced Flux attention call with TACA temperature control for regional rendering.
        
        Args:
            attn: The attention module
            hidden_states: Input hidden states
            encoder_hidden_states: Optional encoder hidden states
            attention_mask: Optional attention mask
            image_rotary_emb: Optional rotary embeddings for images
            base_ratio: Ratio for regional control. TACA is applied only when this is not None.
        
        Returns:
            Processed hidden states with optional encoder hidden states
        """
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        
        # Apply TACA temperature control only during mask injection steps
        apply_taca = self.enable_taca_temp_control and base_ratio is not None
        
        # Sample projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # Handle encoder hidden states for joint attention
        if encoder_hidden_states is not None:
            # Context projections
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
            
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
            
            # Concatenate for joint attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        
        # Apply rotary embeddings if provided
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        
        # Apply attention with optional TACA temperature control
        if apply_taca and HAS_FLEX_ATTENTION:
            # Use flex_attention with TACA score modification
            score_mod = self._create_taca_score_mod()
            hidden_states = flex_attention(query, key, value, score_mod=score_mod)
        elif apply_taca:
            # Fallback to manual TACA scaling with scaled_dot_product_attention
            scale = 1.0 / (head_dim ** 0.5)
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            
            # Apply TACA scaling manually
            query_length = query.shape[-2]
            key_length = key.shape[-2]
            attention_scores = self._apply_manual_taca_scaling(attention_scores, query_length, key_length)
            
            # Apply softmax and compute weighted values
            attention_probs = F.softmax(attention_scores, dim=-1)
            hidden_states = torch.matmul(attention_probs, value)
        else:
            # Standard attention without TACA
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # Handle encoder hidden states output
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            
            # Linear projections and dropout
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            
            return hidden_states, encoder_hidden_states
        else:
            # Linear projection and dropout
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states
    
    def __call__(self, *args, **kwargs):
        """Make the processor callable, delegating to FluxAttnProcessor2_0_call."""
        return self.FluxAttnProcessor2_0_call(*args, **kwargs)