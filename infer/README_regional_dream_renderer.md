# RegionalDreamRendererFluxAttnProcessor2_0 with TACA Integration

This implementation integrates TACA's temperature control mechanism into a Regional Dream Renderer Flux Attention Processor for improved text-image alignment during regional control operations.

## Overview

The `RegionalDreamRendererFluxAttnProcessor2_0` class enhances the standard Flux attention mechanism with TACA's temperature control to improve text-image alignment specifically during mask injection steps in regional control pipelines.

## Features

- **TACA Temperature Control**: Dynamically rebalances cross-modal attention using temperature scaling
- **Conditional Application**: TACA is applied only during mask injection steps (`base_ratio` is not None)
- **Configurable Parameters**: Customizable scale factor and encoder size
- **Dual Implementation**: Uses `flex_attention` when available, fallback to manual scaling otherwise
- **Backward Compatibility**: Maintains existing functionality when TACA is disabled

## Quick Start

```python
from infer.regional_dream_renderer_flux_attn_processor import RegionalDreamRendererFluxAttnProcessor2_0

# Create processor with default TACA settings
processor = RegionalDreamRendererFluxAttnProcessor2_0()

# Or customize TACA parameters
processor = RegionalDreamRendererFluxAttnProcessor2_0(
    taca_scale=1.2,                    # Temperature scale factor
    taca_encoder_size=512,             # Encoder size threshold
    enable_taca_temp_control=True      # Enable/disable TACA
)

# Apply to attention modules in your transformer
for name, module in transformer.named_modules():
    if hasattr(module, 'processor'):
        module.processor = processor
```

## Parameters

### Constructor Parameters

- `taca_scale` (float, default=1.2): Temperature scale factor for TACA attention enhancement
- `taca_encoder_size` (int, default=512): Encoder size threshold that separates text and image tokens
- `enable_taca_temp_control` (bool, default=True): Whether to enable TACA temperature control

### Attention Call Parameters

The processor extends the standard Flux attention call with:

- `base_ratio` (Optional[float]): When provided (not None), indicates a mask injection step and enables TACA

## How TACA Works

TACA (Temperature-Adaptive Cross-modal Attention) improves text-image alignment by:

1. **Identifying Cross-Modal Attention**: Detects when image tokens attend to text tokens
2. **Temperature Scaling**: Applies scaling factor to attention scores in the image-to-text region
3. **Selective Enhancement**: Only affects attention from image tokens (≥ encoder_size) to text tokens (< encoder_size)

### Attention Matrix Regions

```
           Text Tokens    Image Tokens
           [0:512]        [512:]
Text    ┌─────────────┬─────────────┐
Tokens  │             │             │
[0:512] │  Text-Text  │ Text-Image  │
        │             │             │
        ├─────────────┼─────────────┤
Image   │             │             │
Tokens  │ Image-Text  │Image-Image  │
[512:]  │ (SCALED)    │             │
        └─────────────┴─────────────┘
```

Only the "Image-Text" region is scaled by the TACA mechanism.

## Implementation Details

### Flex Attention vs Manual Scaling

**When `torch.nn.attention.flex_attention` is available:**
```python
# Uses optimized flex_attention with score_mod function
hidden_states = flex_attention(query, key, value, score_mod=score_mod)
```

**Fallback (manual scaling):**
```python
# Computes attention manually and applies TACA scaling
attention_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
attention_scores = self._apply_manual_taca_scaling(attention_scores, ...)
attention_probs = F.softmax(attention_scores, dim=-1)
hidden_states = torch.matmul(attention_probs, value)
```

### Conditional TACA Application

```python
# TACA is applied only during mask injection steps
apply_taca = self.enable_taca_temp_control and base_ratio is not None

if apply_taca and HAS_FLEX_ATTENTION:
    # Use flex_attention with TACA
elif apply_taca:
    # Use manual TACA scaling
else:
    # Standard attention without TACA
```

## Testing

Run the comprehensive test suite:

```bash
python test_regional_dream_renderer.py
```

This tests:
- Initialization with different parameters
- Standard vs TACA-enhanced attention
- Output shape consistency
- TACA enable/disable functionality
- Score modification correctness
- Manual scaling fallback

## Example Usage

See `example_usage.py` for a complete demonstration:

```bash
python example_usage.py
```

## Integration with Existing Pipelines

The processor is designed to be a drop-in replacement for standard Flux attention processors:

```python
# Replace existing processor
old_processor = attention_module.processor
attention_module.processor = RegionalDreamRendererFluxAttnProcessor2_0(
    taca_scale=1.2,
    taca_encoder_size=512,
    enable_taca_temp_control=True
)

# TACA will automatically activate during mask injection
# when base_ratio is provided in the attention call
```

## Performance Considerations

- **Flex Attention**: Preferred when available for optimal performance
- **Manual Scaling**: Automatic fallback maintains functionality on all systems
- **Conditional Application**: TACA overhead only occurs during mask injection steps
- **Memory Efficiency**: No additional memory allocation for standard operations

## Compatibility

- **PyTorch**: Requires PyTorch with tensor operations support
- **Diffusers**: Compatible with diffusers library attention modules
- **Flex Attention**: Optional dependency for optimized implementation
- **CUDA**: Supports both CPU and GPU execution

## References

- [TACA Paper](https://arxiv.org/abs/2506.07986/): Original TACA methodology
- [TACA Project Page](https://vchitect.github.io/TACA/): Additional resources and examples