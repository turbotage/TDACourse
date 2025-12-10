import sys
sys.path.append('.')

import torch
from Models import GMMVAE_CNN

# Create model with same config as training
model = GMMVAE_CNN(
    input_shape=(3, 32, 32),
    embedding_dim=32,
    num_classes=10,
    conv_channels=[32, 64, 128, 256],
    use_residual=True,
    num_residual_blocks=1
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

encoder_params = sum(p.numel() for p in model.encoder.parameters())
decoder_params = sum(p.numel() for p in model.decoder.parameters())
gmm_params = total_params - encoder_params - decoder_params

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"\nBreakdown:")
print(f"  Encoder: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
print(f"  Decoder: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
print(f"  GMM (π, μ, σ²): {gmm_params:,} ({gmm_params/total_params*100:.1f}%)")
print(f"\nEncoder/Decoder ratio: {encoder_params/decoder_params:.2f}:1")
