---
title: Spatial Splat
sdk: gradio
sdk_version: 6.15.2
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
models:
  - VAST-AI/TripoSplat
  - WilliamQM/Spatial-Splat-LoRA
---

# Spatial Splat

Private proof-of-concept interface for a paired six-way comparison:

- Base TripoSplat
- Spatial LoRA
- Rank-2 Spatial LoRA compressed from the rank-8 adapter
- Procedural six-view LoRA trained from exact primitive RGB and depth renders
- Primitive geometry control
- Spatial LoRA plus primitive geometry control

Upload a generated image and its matching primitive JSON. The Space accepts
both the canonical Spatial Splat schema and WorldSketch/WorldSplat exports
using `type`, `position`, `rotation`, and `scale`. Editor exports are uniformly
normalized into the isometric frame used during adapter training. Every mode
uses the same encoded image and exact Phase 2 noise. The 3D viewers draw the
primitive wireframe over each Gaussian splat for direct spatial inspection.

Geometry control defaults to the first 70% of flow steps, leaving the final
30% to the normal Phase 2 model for detail cleanup. The `Control ends` slider
sets this cutoff; `1.0` restores the original always-on behavior.
The geometry-control strength slider intentionally extends to `15` for
stress-testing; the trained default remains `1`.

Six bundled templates can fill both inputs in one click. Three new realistic
geometry tests cover a grassy Pine Clearing, a Cactus Desert, and a Wizard
Tower on stony terrain. Three earlier synthetic tests cover Palm Oasis, Garden
Bench, and Cylinder Droid. None of these examples were used to train the LoRA
or geometry-control adapter.

Each run also rasterizes every output from isometric, top, left, right, front,
and back views. The comparison table reports spatial rating, structure loss,
worst P95 normalized depth error, median depth error, silhouette and bounding-
box IoU, centroid and extent errors, plus floor P95 and flatness when a support
surface exists. The complete per-view record is downloadable as JSON.

The geometry adapter samples analytic primitive SDFs at TripoSplat's 8,192
fixed Sobol token positions and injects them into six Phase 2 transformer
blocks at every diffusion step. It is experimental and does not guarantee
correct hidden geometry.

The rank-2 adapter is a per-layer truncated-SVD compression of the rank-8
Spatial LoRA. It retains 87.6% of the trained adapter's matrix-update energy
with one quarter of the LoRA parameters. This tests whether the strongest
learned directions carry spatial correction without the weaker appearance
changes; it does not assume that individual latent coordinates have fixed
transform meanings.

The procedural six-view LoRA is an experimental rank-4 adapter trained on 12
quality-gated primitive scenes. On one paired seed it improved 3 of 4 unseen
procedural scenes, with median structure improvement of 4.5%, but mean
structure loss was 6.0% worse because the held-out cluster regressed. It is
included for visual stress-testing and is not a generalization claim.

On one paired seed across two held-out scenes, Geometry Control improved mean
six-view structure loss from `0.1051` to `0.0942` and mean worst-view P95 depth
from `0.4519` to `0.3971`. This is a small POC result, not a generalization
claim.
