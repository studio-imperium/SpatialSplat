---
license: mit
base_model: VAST-AI/TripoSplat
library_name: pytorch
tags:
  - gaussian-splatting
  - image-to-3d
  - lora
  - controlnet
  - spatial-consistency
---

# Spatial Splat Adapters

Spatial Splat provides three experimental adapters for the Phase 2
image-to-latent flow model in
[VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat):

- A rank-8 spatial LoRA.
- A rank-2 SVD-compressed version of the spatial LoRA.
- A primitive-SDF geometry control adapter.

Phase 1 and the Gaussian decoder remain unchanged. The control adapter samples
analytic primitive geometry at TripoSplat's 8,192 fixed Sobol token positions
and injects zero-initialized residuals into six Phase 2 transformer blocks at
every diffusion step.

`scene_control_tensor` accepts both the canonical Spatial Splat schema and
WorldSketch/WorldSplat primitive exports using `type`, `position`, `rotation`,
and `scale`. Editor-space scenes are uniformly normalized into the isometric
training frame before feature construction.

Try it in the private
[Spatial Splat Space](https://huggingface.co/spaces/WilliamQM/Spatial-Splat).

The LoRA was trained on 18 accepted pseudo-target latents. The geometry adapter
was trained on seven stricter targets and selected against two held-out scenes.
Each target was optimized against primitive geometry from six views. The
objective emphasizes coarse object placement, orientation, scale, bounding
boxes, and support surfaces over small visual details.

## POC result

The frozen test contains five unseen scenes and three paired seeds per scene.

| Metric | Base | Spatial LoRA |
|---|---:|---:|
| Mean structure loss | 0.6572 | 0.5463 |
| Structure win rate | - | 13/15 |
| Mean P95 depth error | 1.2676 | 1.2671 |
| P95 win rate | - | 10/15 |

All five test scenes improved in average structure loss. Worst-view geometry
is still inconsistent, so this checkpoint should be treated as a proof of
concept rather than a production spatial-control model.

The geometry adapter reduced the held-out base latent-flow probe loss from
`0.6827` to `0.5907` (13.5%). With the LoRA enabled, the probe remained
effectively flat (`0.5769` to `0.5770`). A shuffled-scene control scored almost
the same as the correct control (`0.5908`), so this first small-data checkpoint
has not yet demonstrated strong scene-specific control. The five-way Space is
provided to inspect that limitation directly.

One paired fresh-generation seed on the two held-out geometry scenes produced:

| Metric | Base | LoRA | Geometry control | Combined |
|---|---:|---:|---:|---:|
| Mean structure loss | 0.1051 | 0.1183 | **0.0942** | 0.0988 |
| Mean worst-view P95 depth | 0.4519 | 0.5059 | **0.3971** | 0.4448 |
| Mean minimum soft IoU | 0.8629 | 0.8583 | **0.8758** | 0.8737 |

Geometry control beat Base on both scenes. This is encouraging but far too
small an evaluation to establish generalization; use the Space to test new
images and matching primitive contracts.

## Files

- `flow_lora.safetensors`: 3,670,016 trainable parameters.
- `flow_lora_config.json`: target modules and adapter settings.
- `flow_lora_rank2.safetensors`: 917,504 parameters and 87.6% retained
  matrix-update energy.
- `flow_lora_rank2_config.json`: rank-2 compression metadata.
- `spatial_lora.py`: minimal loader for the original TripoSplat flow model.
- `spatial_control.safetensors`: primitive geometry control weights.
- `spatial_control_config.json`: feature schema and injection settings.
- `spatial_control.py`: SDF feature builder and adapter loader.
- `model.py` and `triposplat.py`: control-enabled TripoSplat Phase 2 path.

## Usage

```python
import torch

from triposplat import TripoSplatPipeline
from spatial_control import load_spatial_control, scene_control_tensor
from spatial_lora import load_spatial_lora

pipe = TripoSplatPipeline(
    ckpt_path="ckpts/diffusion_models/triposplat_fp16.safetensors",
    decoder_path="ckpts/vae/triposplat_vae_decoder_fp16.safetensors",
    dinov3_path="ckpts/clip_vision/dino_v3_vit_h.safetensors",
    flux2_vae_encoder_path="ckpts/vae/flux2-vae.safetensors",
    rmbg_path="ckpts/background_removal/birefnet.safetensors",
    device="cuda",
)
load_spatial_lora(
    pipe.flow_model,
    "flow_lora.safetensors",
    "flow_lora_config.json",
)
load_spatial_control(
    pipe.flow_model,
    "spatial_control.safetensors",
    "spatial_control_config.json",
    device="cuda",
)
generator = torch.Generator(device="cuda").manual_seed(42)
prepared = pipe.preprocess_image("image.png")
condition = pipe.encode_image(prepared, generator=generator)
control = scene_control_tensor("scene.json", device="cuda")
sample = pipe.sample_latent(
    condition,
    generator=generator,
    control=control,
    control_scale=1.0,
)
```

These are custom TripoSplat adapters, not PEFT or Diffusers checkpoints.

## LoRA training

- Rank and alpha: 8
- Steps: 800
- Learning rate: 5e-5
- Accepted targets: 18
- Supervision: six-view depth, silhouette, centroid, extent, support coverage,
  and support flatness

## Geometry-control training

- Trainable parameters: 1,209,600
- Steps: 600
- Learning rate: 1e-4
- Accepted targets: seven train and two validation
- Inputs: analytic primitive SDF, occupancy, surface bands, normals, primitive
  kind, and support-surface flags at 8,192 fixed Sobol points
- Injection points: six Phase 2 transformer blocks at every flow step

The control adapter keeps the base model and LoRA frozen, alternates LoRA-on
and LoRA-off training steps, and selects its deployable checkpoint using
held-out validation controls.

## License and attribution

Released under the MIT license. TripoSplat code and weights are also MIT
licensed. Please cite the original TripoSplat project when using this adapter.
