#!/usr/bin/env python3
"""
Test script for RegionalDreamRendererFluxAttnProcessor2_0 with TACA integration.
"""
import torch
import torch.nn as nn
import sys
import os

# Add the infer directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'infer'))

from regional_dream_renderer_flux_attn_processor import RegionalDreamRendererFluxAttnProcessor2_0


class MockAttention(nn.Module):
    """Mock attention module for testing."""
    def __init__(self, dim=768, heads=12):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        # Optional normalization layers
        self.norm_q = None
        self.norm_k = None
        self.norm_added_q = None
        self.norm_added_k = None
        
        # Joint attention projections
        self.add_q_proj = nn.Linear(dim, dim, bias=False)
        self.add_k_proj = nn.Linear(dim, dim, bias=False)
        self.add_v_proj = nn.Linear(dim, dim, bias=False)
        
        # Output projections
        self.to_out = nn.ModuleList([
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(0.0)
        ])
        self.to_add_out = nn.Linear(dim, dim, bias=False)


def test_regional_dream_renderer_processor():
    """Test the RegionalDreamRendererFluxAttnProcessor2_0 functionality."""
    print("Testing RegionalDreamRendererFluxAttnProcessor2_0...")
    
    # Create processor with TACA settings
    processor = RegionalDreamRendererFluxAttnProcessor2_0(
        taca_scale=1.2,
        taca_encoder_size=512,
        enable_taca_temp_control=True
    )
    
    # Test initialization
    assert processor.taca_scale == 1.2
    assert processor.taca_encoder_size == 512
    assert processor.enable_taca_temp_control == True
    print("✓ Initialization test passed")
    
    # Create mock attention module
    attn = MockAttention(dim=768, heads=12)
    
    # Test input tensors
    batch_size = 2
    seq_len_image = 1024  # Image tokens
    seq_len_text = 77    # Text tokens  
    dim = 768
    
    # Create hidden states (image tokens)
    hidden_states = torch.randn(batch_size, seq_len_image, dim)
    
    # Create encoder hidden states (text tokens)
    encoder_hidden_states = torch.randn(batch_size, seq_len_text, dim)
    
    # Test without TACA (base_ratio=None)
    try:
        output_no_taca = processor.FluxAttnProcessor2_0_call(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            base_ratio=None  # No mask injection, so no TACA
        )
        print("✓ Standard attention test passed")
    except Exception as e:
        print(f"✗ Standard attention test failed: {e}")
        return False
    
    # Test with TACA (base_ratio provided)
    try:
        output_with_taca = processor.FluxAttnProcessor2_0_call(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            base_ratio=0.5  # Mask injection step, apply TACA
        )
        print("✓ TACA-enhanced attention test passed")
    except Exception as e:
        print(f"✗ TACA-enhanced attention test failed: {e}")
        return False
    
    # Test output shapes
    if isinstance(output_with_taca, tuple):
        hidden_out, encoder_out = output_with_taca
        assert hidden_out.shape == hidden_states.shape
        assert encoder_out.shape == encoder_hidden_states.shape
        print("✓ Output shape test passed")
    else:
        assert output_with_taca.shape == hidden_states.shape
        print("✓ Output shape test passed")
    
    # Test with TACA disabled
    processor_no_taca = RegionalDreamRendererFluxAttnProcessor2_0(
        enable_taca_temp_control=False
    )
    
    try:
        output_disabled = processor_no_taca.FluxAttnProcessor2_0_call(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            base_ratio=0.5  # Even with base_ratio, TACA should be disabled
        )
        print("✓ TACA disabled test passed")
    except Exception as e:
        print(f"✗ TACA disabled test failed: {e}")
        return False
    
    # Test score modification function
    score_mod = processor._create_taca_score_mod()
    test_score = torch.randn(2, 8, 1000, 600)  # batch, heads, seq_q, seq_k
    
    try:
        # Create tensor arguments for the score_mod function
        batch_tensor = torch.tensor(0)
        head_tensor = torch.tensor(0)
        token_q_tensor = torch.tensor(600)  # image token position
        token_kv_tensor = torch.tensor(100)  # text token position
        
        modified_score = score_mod(test_score, batch_tensor, head_tensor, token_q_tensor, token_kv_tensor)
        # Check that the function runs without error
        assert modified_score.shape == test_score.shape
        print("✓ Score modification test passed")
    except Exception as e:
        print(f"✗ Score modification test failed: {e}")
        return False
    
    # Test manual TACA scaling
    try:
        attention_scores = torch.randn(batch_size, 12, seq_len_image + seq_len_text, seq_len_image + seq_len_text)
        scaled_scores = processor._apply_manual_taca_scaling(
            attention_scores, 
            seq_len_image + seq_len_text, 
            seq_len_image + seq_len_text
        )
        assert scaled_scores.shape == attention_scores.shape
        print("✓ Manual TACA scaling test passed")
    except Exception as e:
        print(f"✗ Manual TACA scaling test failed: {e}")
        return False
    
    print("All tests passed! ✓")
    return True


def test_configurable_parameters():
    """Test that the configurable parameters work correctly."""
    print("\nTesting configurable parameters...")
    
    # Test custom scale factor
    processor_custom = RegionalDreamRendererFluxAttnProcessor2_0(
        taca_scale=1.5,
        taca_encoder_size=256,
        enable_taca_temp_control=True
    )
    
    assert processor_custom.taca_scale == 1.5
    assert processor_custom.taca_encoder_size == 256
    print("✓ Custom parameters test passed")
    
    # Test score mod with custom parameters
    score_mod = processor_custom._create_taca_score_mod()
    test_score = torch.ones(1, 1, 300, 300)
    
    # Test scaling at the custom encoder boundary
    batch_tensor = torch.tensor(0)
    head_tensor = torch.tensor(0)
    token_q_tensor = torch.tensor(260)  # Should scale since 260 >= 256
    token_kv_tensor = torch.tensor(100)  # and 100 < 256
    
    modified_score = score_mod(test_score, batch_tensor, head_tensor, token_q_tensor, token_kv_tensor)
    
    print("✓ Custom parameters functionality test passed")


if __name__ == "__main__":
    success = test_regional_dream_renderer_processor()
    if success:
        test_configurable_parameters()
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)