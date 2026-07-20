# Spatial Splat Hugging Face Deployment

This directory contains the two private deployment packages:

- `model`: deployed to `WilliamQM/Spatial-Splat-LoRA`.
- `space`: deployed to `WilliamQM/Spatial-Splat` on ZeroGPU.

The Space includes the LoRA and primitive-SDF control adapter, then downloads the public
`VAST-AI/TripoSplat` base checkpoints during startup. It never uploads or
duplicates the original base weights.

## Hosted smoke test

The deployed Space was tested with the Cylinder Droid input at seed 42, 20
steps, guidance 3.0, and 32,768 Gaussians:

- Spatial LoRA: 15.0 seconds.
- Base TripoSplat: 13.7 seconds.
- Both returned valid 2.1 MB PLY files with different SHA-256 hashes.

The current Space runs Base, rank-8 LoRA, rank-2 compressed LoRA, Geometry
Control, and rank-8 LoRA plus Geometry Control from paired noise. It overlays
the uploaded primitive scene on every result.
