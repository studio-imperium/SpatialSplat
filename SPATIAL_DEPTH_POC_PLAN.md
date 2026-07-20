# SpatialSplat Multi-View Depth POC Plan

## Status

This document describes the first intentionally small proof of concept for
spatially supervised TripoSplat training. It is also an engineering handoff for
agents working in this repository.

The POC asks one question:

> Can a rich image generated from a primitive layout be mapped by TripoSplat to
> a Gaussian splat whose coarse visible geometry agrees with the original
> primitives from fixed isometric, top, left, right, front, and back viewpoints?

The primitive scene is not the visual target. It is a coarse spatial contract.
Generated texture, decoration, and small geometric details are allowed and
should contribute little or nothing to the spatial score.

## Implementation Progress (2026-07-18)

Implemented locally:

- `training.scene_schema`: deterministic primitive and orthographic-camera
  artifact schema.
- `training.primitive_renderer`: analytic box, sphere, cylinder, and platform
  rendering with exact RGB, depth, alpha, boundary, and visible instance IDs.
- `training.generate_poc_data`: deterministic generation of all ten POC scenes
  and their image-generation prompts.
- Ten rich generated condition images under the ignored local `poc_data/`
  workspace directory.
- `training.spatial_loss`: NumPy evaluation implementation of the coarse,
  tolerant depth/mask/centroid/extent objective.
- `training.spatial_loss_torch`: matching differentiable PyTorch objective for
  Phase 3.
- `training.gsplat_depth_renderer`: CUDA expected-depth/alpha rasterization
  through `gsplat`, including the calibrated raw-decoder coordinate transform.
- `training.decode_train`: frozen, fixed-anchor Gaussian decoder path with
  gradients to latent tokens and optional activation checkpointing.
- `training.optimize_one_latent`: staged image encoding, Phase 2 sampling,
  six-view latent-only optimization, and fixed/fresh-anchor evaluation.
- `training.evaluate_lora_generation`: resumable paired base/LoRA normal
  generation from fresh flow noise, normal Phase 1 decoding, six-view scoring,
  and complete PLY/depth/alpha artifact export.
- `training.multiview`: deterministic isometric, top, left, right, front, and
  back supervision
  cameras plus exact primitive depth, mask, and boundary targets.
- `training.validate_gsplat_renderer`: CPU/CUDA renderer parity gate.
- `training.evaluate_spatial`: command-line metric evaluator.
- `training.generate_baselines`: A6000 batch runner for cached image features,
  base latent/camera samples, and low-budget baseline splats.
- `training.generate_remote_baselines`: resumable client for the hosted Gradio
  TripoSplat service used to generate the ten 32,768-Gaussian baseline PLYs.
- `training.gaussian_depth_renderer`: CPU Gaussian alpha/depth renderer using
  the exact POC isometric camera, Gaussian covariance, and front-to-back alpha
  compositing.
- `training.render_baseline_depths`: batch depth rendering and primitive-edge
  overlay generation for downloaded PLYs.
- `training.score_baselines`: per-scene metrics plus an aggregate baseline
  summary.
- Unit tests for primitive rendering, score behavior, NumPy/Torch agreement,
  finite gradients to predicted depth and alpha, multi-view cameras, and the
  pseudo-target quality gate.

Verified score behavior on the center-cube scene:

```text
exact prediction:             0.0000 loss
sub-tolerance depth change:   0.0000 loss
48-pixel horizontal shift:    0.2725 loss
empty prediction:             1.7324 loss
```

Completed without a local GPU:

- Generated all ten baseline splats through `http://148.153.245.160:17860/`
  using seed 42, 20 steps, guidance 3.0, and 32,768 Gaussians.
- Calibrated one global model-to-primitive transform: scale `0.96` and vertical
  translation `-0.25`. No per-scene camera or geometry fitting is used.
- Rendered baseline alpha/depth from the exact isometric camera and generated
  red primitive / cyan splat alignment overlays.

Completed on a RunPod RTX A6000:

- Validated the CUDA renderer against the saved CPU baseline. Spatial loss was
  `0.1265` with `gsplat` versus `0.1367` with the CPU evaluator; mean alpha
  difference was `0.0033`.
- Verified a finite spatial-loss gradient to the latent (`0.2895` norm).
- Optimized the center-cube latent for 60 steps at 256 x 256 and evaluated at
  512 x 512 with a completely fresh octree anchor sample.
- Fresh-anchor spatial loss improved from `0.1252` to `0.00839` (`93.3%`), and
  soft mask IoU improved from `0.8509` to `0.9884`.
- The optimization itself took `13.8` seconds after model/setup caches were
  ready and used `2.61` GiB peak VRAM. The full first RunPod session, including
  setup, checkpoint download, CUDA compilation, tests, and evaluation, ran for
  about 11 minutes at `$0.33/hour`.
- Saved `target_latent.safetensors`, `optimized_splat.ply`, metrics, depth,
  alpha, overlays, and per-step history under `poc_data/01_center_cube/`.

Completed on a RunPod A40:

- Optimized the remaining nine scenes with the same 60-step latent-only loop.
- All ten targets passed fresh-anchor acceptance. Mean spatial loss improved
  from `0.2091` to `0.09694` (`66.3%` mean relative improvement), and mean soft
  mask IoU improved from `0.7995` to `0.9209`.
- Fresh-anchor relative improvements range from `33.9%` on the hardest
  three-step layout to `93.3%` on the center cube.
- Visually audited every textured optimized PLY with exact primitive wireframes
  in `poc_viewer.html`. All ten remain recognizable without opacity or Gaussian
  collapse; the three-step and diagonal-pair scenes retain visible residual
  shape error.
- Packaged all ten accepted image/conditioning/base-camera/target-latent pairs
  in `poc_data/lora_dataset.json`.
- Trained rank-8 LoRA adapters on 112 Phase 2 attention/MLP linear layers for
  1,000 flow-matching steps. This updates `3,670,016` adapter parameters while
  the original Phase 2 and all of Phase 1 stay frozen.
- Fixed-probe flow loss improved from `0.52179` to `0.47265` (`9.4%`) with
  `2.81` GiB peak VRAM. The final 15 MB adapter, ten 100-step checkpoints,
  history, configuration, and summary are under `poc_data/lora_run/`.

This first adapter is retained as a diagnostic artifact, but it is not the
clean training result. Visual review found three flipped or rotated targets,
and the original improvement-only acceptance gate allowed them into training.

Multi-view correction run:

- Added equal-weight isometric, top, and side depth/alpha supervision to every
  latent-optimization step while keeping Phase 1 frozen.
- Added fresh-anchor per-view metrics and a strict pseudo-target gate. A target
  must improve, keep IoU from regressing, have worst-view soft IoU at least
  `0.80`, worst-view median normalized depth error at most `0.15`, and
  worst-view P95 normalized depth error at most `0.20`.
- The P95 gate rejects the known flipped scenes from the original isometric
  run: Cube Sphere, Three Step, and Diagonal Pair.
- The full three-view run revealed hidden geometry errors that the isometric
  viewer could not show. Four targets currently pass all views: Center Cube,
  Center Sphere, Front/Back, and Pillars.
- A replacement LoRA is trained only from these clean targets and is written to
  `poc_data/lora_run_multiview/`; its manifest is
  `poc_data/lora_dataset_multiview.json`.
- The clean rank-8 LoRA completed 1,000 steps over four accepted pairs in
  `1,969` seconds with `2.68` GiB peak VRAM. Fixed-probe flow loss improved
  from `0.43729` to `0.39577` (`9.49%`). The final adapter and ten 100-step
  checkpoints are downloaded locally.

Six-view extension:

- Expanded optimization and fresh-anchor gating to isometric, top, left,
  right, front, and back. The three-view adapter remains as a comparison
  artifact.
- Re-optimized all ten latents for 60 steps. Center Sphere and Pillars narrowly
  missed the gate, then passed after a targeted 180-step retry. The final clean
  set remains Center Cube (`0.161` worst-view P95), Center Sphere (`0.135`),
  Front/Back (`0.173`), and Pillars (`0.191`).
- Trained a separate rank-8 LoRA for 1,000 steps from those four six-view
  targets. Fixed-probe flow loss improved from `0.44628` to `0.40218`
  (`9.88%`) with `2.68` GiB peak VRAM.
- The six-view manifest is `poc_data/lora_dataset_six_view.json`. The final
  adapter and ten checkpoints are under `poc_data/lora_run_six_view/`.
- Evaluation-only camera checks are not treated as training supervision; all
  six cameras participate in the latent-optimization gradient.

Fresh-generation stress test:

- Evaluated base and six-view LoRA normal generation with paired flow-noise
  seeds `101`, `202`, and `303` on all four accepted scenes. Each pair used the
  same cached image conditioning, initial noise, and a newly sampled matching
  octree seed before six-view scoring.
- The LoRA beat base on all 12 pairs. Mean spatial loss fell from `0.26827` to
  `0.11836`, a `52.0%` relative improvement.
- The LoRA recovered `64.5%` of the optimized-target improvement on average,
  exceeding the predeclared `50%` Phase 4 criterion.
- Per-scene mean spatial-loss improvements were Center Cube `70.0%`, Center
  Sphere `48.8%`, Front/Back `65.9%`, and Pillars `34.5%`.
- This is a **Phase 4 POC pass for trainability/memorization**. It is not yet a
  generalization result because these are fresh samples of the four training
  conditions, not unseen primitive layouts.
- Worst-view depth tails remain imperfect. Mean aggregate P95 improved on three
  scenes, but Center Sphere changed from `1.043` to `1.051`, and most generated
  samples remain above the strict `0.20` pseudo-target gate. The adapter has
  learned the broad geometry far better without eliminating every local depth
  outlier.
- Complete results and visual artifacts are under
  `poc_data/fresh_generation_six_view/` and can be inspected in
  `poc_viewer.html` as Fresh base versus Fresh LoRA.

Held-out generalization test:

- Added three scenes under `poc_data/heldout/` that were never used for latent
  optimization or LoRA training: Archway, Raised Table, and Diagonal
  Procession. Each has a new generated condition image and exact six-view
  primitive targets.
- The first held-out run was preprocessing-confounded: BiRefNet removed the
  entire platform from Raised Table and inconsistently removed spatial context
  from the other RGB inputs, while the exact targets still included it. Those
  diagnostic artifacts remain under `poc_data/heldout_generation_six_view/`
  but are not the final generalization result.
- Added alpha-safe input handling. `training.generate_baselines` can now require
  real alpha and records whether background removal ran; held-out inputs use
  `--require-alpha --erode-radius 0`. The corrected prepared images preserve
  the complete platform and skip BiRefNet.
- Re-evaluated three paired fresh-noise seeds per scene. The LoRA won `8/9`
  pairs. Mean spatial loss changed from `0.40890` to `0.31027`, a `15.27%`
  average improvement, and median pair improvement was `7.19%`.
- Archway improved `39.4%` but retained one bad seed, Raised Table improved
  `12.3%` and won all three seeds, and Diagonal Procession improved `4.6%` and
  won all three seeds.
- This is **early positive generalization evidence**, not a final pass. Only
  three layouts were tested, absolute P95 tails remain high, and Archway is
  still seed-sensitive. The result justifies scaling target diversity before
  changing architecture.
- Corrected artifacts are under
  `poc_data/heldout_alpha_generation_six_view/` and are selectable in
  `poc_viewer.html` as Fresh base versus Fresh LoRA.

Structural scoring correction for the scaled dataset:

- Do not let a large correct floor/platform hide a wrong object orientation.
  Score support surfaces and non-support objects as separate channels using
  `primitive_ids.npy`; the object channel excludes the ground primitive.
- Treat object position, scale, and orientation as hard gates rather than only
  terms in an average. Each view must pass object-mask bounding-box IoU,
  centroid, and extent checks. An asymmetric depth/occupancy signature across
  the six views must reject mirrored or 180-degree-wrong layouts.
- Score floors and table-like supports separately using visible height error,
  plane residual/flatness, and P95 depth error. These terms should strongly
  penalize warped surfaces without requiring generated decorative detail to
  match the primitive proxy.
- Aggregate views with a worst-view or upper-tail term in addition to the mean.
  A candidate that fails any orientation/placement gate is excluded from LoRA
  training even if its overall spatial score looks acceptable.

Diverse 50-scene scaffold:

- Added `training.generate_diverse_data` and generated 50 new primitive
  contracts under `poc_data/diverse_train/` without consuming image-generation
  quota.
- The split contains 35 grounded scenes across woodland, desert, rock, snow,
  ruins, gardens, and industry, plus 15 floorless robots, ordinary objects, and
  abstract structures.
- Each scene includes exact primitive RGB/depth/mask/boundary artifacts, a
  material-aware image prompt, a chroma-key choice, and a condition spec that
  requires real alpha.
- Added `training.add_condition_alpha` for preserving existing rendered RGB
  pixels while removing only border-connected neutral backgrounds.
- Added `training.generate_controlnet_images`, a separate SDXL bulk-generation
  harness using exact primitive depth plus boundary ControlNets. It records
  model IDs, seeds, control strengths, runtime, and alpha coverage. The default
  command is a ten-scene diversity pilot; `--all` expands it to all 50 after
  visual review.
- Generated all 50 rich condition images with SDXL img2img plus depth and
  boundary ControlNets. The primitive RGB proxy is placed on white as the
  img2img start, so texture can change while orientation, object positions,
  scale, bounding boxes, and support heights remain controlled throughout
  denoising.
- Default generation settings are 768 x 768, seed sequence starting at 24000,
  40 steps, strength 0.95, depth scale 0.95, and boundary scale 0.85. The one
  initial outlier (`04_fallen_log`) invented a large plume and was regenerated
  successfully at strength 0.82.
- RGBA uses border-connected neutral-background removal plus the exact control
  subject mask as a protected minimum. This prevents white snow/floor/table
  surfaces from disappearing while still retaining generated detail outside
  the primitive silhouette.
- The final audit has 50/50 RGB, RGBA, and metadata records. Visual review found
  no remaining wrong orientations or missing support surfaces. The largest
  excess alpha footprint is 3.74% over its target (`37_robot_rover`); all other
  scenes are lower. Results and `generation_audit.json` are under
  `poc_data/diverse_train/`.
- On the cached A40, generation took about 7.7-8.2 seconds per image. The full
  setup, pilot, 50-image run, correction, download, and audit session ran for
  29m43s, about $0.218 of GPU compute at $0.44/hour. The stopped pod retains
  `/root/imagegen-venv` and `/root/hf-cache` for later runs.

Pending POC work:

- Increase pseudo-target diversity before claiming spatial generalization.

Local setup and verification:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-poc.txt
# Install torch separately for the current platform.
.venv/bin/python -m pytest -q
.venv/bin/python -m training.generate_poc_data --output poc_data --resolution 512
# On the CUDA machine after installing the matching torch build:
.venv/bin/python -m training.generate_baselines --data-root poc_data --checkpoint-root ckpts
.venv/bin/python -m training.validate_gsplat_renderer --scene-dir poc_data/01_center_cube
.venv/bin/python -m training.optimize_one_latent \
  --scene-dir poc_data/01_center_cube \
  --optimization-steps 60 --render-size 256 --final-render-size 512 \
  --views isometric top left right front back
.venv/bin/python -m training.optimize_latent_batch \
  --data-root poc_data --checkpoint-root ckpts
.venv/bin/python -m training.build_lora_dataset \
  --data-root poc_data --output poc_data/lora_dataset_six_view.json \
  --max-p95-depth-error 0.20 --allow-rejected
.venv/bin/python -m training.train_flow_lora \
  --manifest poc_data/lora_dataset_six_view.json --checkpoint-root ckpts \
  --output-dir poc_data/lora_run_six_view --steps 1000
.venv/bin/python -m training.evaluate_lora_generation \
  --manifest poc_data/lora_dataset_six_view.json --checkpoint-root ckpts \
  --lora poc_data/lora_run_six_view/flow_lora.safetensors \
  --output-dir poc_data/fresh_generation_six_view \
  --seed 101 --seed 202 --seed 303
.venv/bin/python -m training.generate_poc_data \
  --output poc_data/heldout --split heldout --resolution 512
# Add heldout/*/generated_image.png, then on the CUDA machine:
.venv/bin/python -m training.generate_baselines \
  --data-root poc_data/heldout_alpha --checkpoint-root ckpts \
  --require-alpha --erode-radius 0
.venv/bin/python -m training.evaluate_lora_generation \
  --data-root poc_data/heldout_alpha --checkpoint-root ckpts \
  --lora poc_data/lora_run_six_view/flow_lora.safetensors \
  --output-dir poc_data/heldout_alpha_generation_six_view \
  --seed 101 --seed 202 --seed 303
.venv/bin/python -m training.generate_diverse_data \
  --output poc_data/diverse_train --resolution 512
# On the CUDA image-generation environment:
.venv/bin/python -m training.generate_controlnet_images
# Expand only after the ten-scene visual audit:
.venv/bin/python -m training.generate_controlnet_images --all
# Rebuild transparent outputs without loading diffusion weights:
.venv/bin/python -m training.generate_controlnet_images --all --alpha-only
# Current remote baseline path, which does not require local CUDA:
.venv/bin/python -m training.generate_remote_baselines --data-root poc_data
.venv/bin/python -m training.render_baseline_depths --data-root poc_data
.venv/bin/python -m training.score_baselines --data-root poc_data
```

## POC Data Flow

```text
Primitive scene
    -> RGB/depth/boundary controls
    -> rich generated image
    -> TripoSplat image encoders
    -> Phase 2 latent flow model
    -> latent tokens
    -> frozen Phase 1 Gaussian decoder
    -> Gaussian splat
    -> depth and alpha from fixed isometric, top, left, right, front, and back cameras
    -> mean spatial loss plus per-view quality gates against exact primitives
```

The target primitive depth is deterministic. A Gaussian renderer must combine
the translucent splats along each camera ray into one predicted depth map. That
rendering convention is part of the prediction, not uncertainty in the target.

## Repository Context

This repository is currently a compact inference-only implementation. There is
no dataset layer, optimizer, training loop, differentiable Gaussian renderer,
LoRA implementation, checkpoint manager, or evaluation harness.

Important files:

- `triposplat.py`: component loading, image preprocessing and encoding, flow
  sampling, latent decoding, and the public inference pipeline.
- `model.py`: DINOv3, FLUX.2 VAE encoder, background removal, latent flow model,
  octree density decoder, and Gaussian attribute decoder.
- `run_example.py`: basic command-line inference example.
- `run_gradio.py`: inference demo.

Relevant model interfaces:

- `triposplat.FLOW_MODEL_ARGS` configures Phase 2 as an 8192-token, 16-channel
  rectified-flow transformer with a 5-channel camera token, width 1024, and 24
  main transformer blocks.
- `model.LatentSeqMMFlowModel` maps noisy latent and camera tokens plus image
  features to flow velocity predictions.
- `model.OctreeProbabilityFixedlenDecoder` predicts an octree density over
  anchor positions.
- `model.ElasticGaussianFixedlenDecoder` predicts 32 local Gaussians per sampled
  anchor, including offsets, color, scale, rotation, and opacity.
- `model.OctreeGaussianDecoder.decode` performs hard octree sampling and then
  predicts Gaussian attributes.

The paper describes normal Phase 2 training as flow matching against latent
tokens produced by a training-time 3D encoder. That encoder and authoritative
target-token generation pipeline are not present in the released repository.

Paper: <https://arxiv.org/abs/2605.16355>

## What We Can Do

- Use the released image encoders and Phase 2 flow model.
- Expose and cache image conditioning features.
- Sample base latent and camera tokens from the pretrained model.
- Keep every Phase 1 parameter frozen while differentiating through selected
  decoder operations with respect to latent tokens.
- Query the Gaussian attribute decoder directly with a fixed set of anchors.
- Add an external differentiable Gaussian depth/alpha renderer.
- Optimize latent tokens directly from spatial loss.
- Use improved latents as pseudo-targets for ordinary flow-matching LoRA
  training of Phase 2.
- Compare baseline, optimized-latent, and LoRA outputs using exact primitive
  depth and masks.

## What We Cannot Assume

- `OctreeGaussianDecoder.decode` is not differentiable as currently written. It
  is decorated with `torch.no_grad`, and its octree sampling includes discrete
  multinomial/count allocation and masking operations.
- Backpropagation through hard octree sampling will not work by merely removing
  `torch.no_grad`.
- The released model does not provide authoritative latent targets for new GLB
  or primitive scenes.
- Six fixed depth views constrain substantially more geometry than the first
  one-view experiment, but they still do not provide complete 3D supervision.
- Fine appearance cannot be learned from primitive depth. Appearance should be
  preserved from the pretrained model and the rich generated input.
- Ten examples cannot demonstrate generalization. This is a memorization and
  trainability stress test.
- A generated image may violate its primitive controls. Grossly inconsistent
  generated images should be regenerated or excluded, because otherwise the
  image condition and spatial target become contradictory.

## Non-Goals

The first POC does not include:

- ControlNet or direct GLB conditioning inside TripoSplat.
- Full 3D SDF supervision.
- Phase 1 weight updates.
- Full Phase 2 fine-tuning.
- Texture or RGB reconstruction loss.
- A production web application.
- Claims of held-out generalization.

## Phase 1: Generate the Ten Examples

Create ten simple primitive arrangements. Use cubes, spheres, cylinders, a
bounded ground/platform, and a small number of combinations. Use one fixed
isometric camera for every example.

All geometry must be normalized into the coordinate system expected by the
decoder. Record the normalization transform explicitly; do not recover scale
later by fitting each generated splat independently.

For each scene, produce:

```text
scene.json                 primitive types and transforms
camera.json                exact intrinsics and world/camera transforms
primitive_control.png      colored proxy render
primitive_depth.npy        exact metric camera-space depth
primitive_mask.png         exact foreground mask
primitive_boundary.png     optional generation control
generated_image.png        rich image generated from the controls
```

Then run base TripoSplat with a fixed seed and save:

```text
conditioning.safetensors   cached DINOv3 and FLUX.2 image features
base_latent.safetensors    sampled 8192 x 16 latent
base_camera.safetensors    sampled 1 x 5 camera token
base_splat.ply             normal decoded result
base_depth.npy             splat depth from the scoring camera
base_alpha.npy             splat alpha from the scoring camera
base_metrics.json          baseline spatial metrics
```

### Camera Alignment Gate

The primitive target and generated splat must be evaluated in a common frame.
Use either the model's recovered conditioning-view camera once its 5D encoding
has been implemented, or one globally calibrated model-to-primitive transform.
Do not optimize an independent camera or similarity transform for each sample
during final scoring; doing so would conceal pose and scale errors.

Before continuing, verify the primitive wireframe overlays the expected camera
view and that a deliberately translated primitive produces the expected depth
and mask error.

## Phase 2: Implement the Coarse Spatial Score

Add a differentiable Gaussian rasterizer that returns alpha and a single depth
per pixel. During optimization, use alpha/transmittance-weighted depth because
it supplies useful gradients through overlapping translucent Gaussians:

```text
D_splat(p) = sum_i(w_i(p) * z_i) / (sum_i(w_i(p)) + epsilon)
```

Here `z_i` is camera-space depth and `w_i` is the normal front-to-back Gaussian
rendering contribution. The primitive target `D_target` remains exact.

For strict evaluation, also report a first-visible or median-style depth, such
as the depth where accumulated opacity crosses a fixed threshold. Do not train
only against a hard threshold because it provides poor gradients.

### Deliberately Coarse Comparison

Blur and downsample primitive and splat depth/masks to 64 x 64 before scoring.
This suppresses GPT-generated decorations and small surface fluctuations.

Use a dead zone instead of a literal logarithmic near-perfect reward. A literal
log can accidentally increase sensitivity near zero. The intended behavior is
zero or negligible gradient after the coarse geometry is close enough:

```text
e_depth = abs(D_splat_coarse - D_target_coarse) / scene_depth_range
rho(e) = softplus((e - depth_tolerance) / softness) * softness
```

Recommended POC terms:

```text
L_depth    = mean(rho(e_depth) over valid overlapping foreground)
L_mask     = 1 - soft_IoU(A_splat_coarse, M_target_coarse)
L_centroid = normalized distance between foreground centroids
L_extent   = normalized difference in foreground width and height

L_spatial =
    lambda_depth    * L_depth
  + lambda_mask     * L_mask
  + lambda_centroid * L_centroid
  + lambda_extent   * L_extent
```

The mask, centroid, and extent terms prevent undefined or deceptively low depth
loss when the generated splat is missing, transparent, or badly displaced.

For reporting only, convert loss to an intuitive score if desired:

```text
spatial_score = exp(-L_spatial)
```

Optimization should minimize `L_spatial` directly. Reinforcement learning is
not needed for this POC because the selected path is differentiable.

### Score Sanity Tests

The score must rank these perturbations correctly before training:

1. Exact primitive render: best score.
2. Small sub-tolerance translation: almost unchanged.
3. Large translation or scale error: clearly worse.
4. Missing foreground: clearly worse.
5. Added fine decoration inside the silhouette: almost unchanged.
6. Large wall or mass in the wrong place: clearly worse.

## Phase 3: Optimize Latent Tokens

This phase proves that spatial loss can steer a useful part of the pretrained
latent space while Phase 1 remains frozen.

### Differentiable Decoder Path

For the first implementation:

1. Sample octree anchors from the current latent under `torch.no_grad`.
2. Detach and hold those anchors fixed for a short optimization interval.
3. Call `decoder.gs(x=points_pred, cond=latent)` with autograd enabled.
4. Build Gaussian centers and attributes without detaching decoder output.
5. Render differentiable depth and alpha.
6. Backpropagate `L_spatial` into the latent only.
7. Periodically resample anchors from the updated latent, then continue.

This gives gradients through local Gaussian offsets and attributes but not
through the discrete anchor draw itself. Periodic resampling makes the process
piecewise responsive to the octree density without modifying Phase 1 training.

If this path cannot improve a freshly decoded splat, the next escalation is to
add the paper's differentiable octree surface cross-entropy using primitive
surface samples. Do not begin with policy gradients or end-to-end gradients
through the hard sampler.

### Latent Objective

Start from the base sampled latent `z_base` and optimize a copy `z`:

```text
L_prior = mean((z - z_base)^2)
L_total = L_spatial + lambda_prior * L_prior
```

The prior keeps the latent near the pretrained manifold and discourages the
optimizer from destroying appearance or exploiting decoder pathologies.

Suggested initial settings are deliberately conservative:

- Adam over latent only.
- 100 to 300 steps.
- One example at a time.
- Six fixed scoring cameras with equal loss weight.
- Anchor resampling every 10 to 25 steps.
- Save the lowest-loss latent, not merely the final latent.

### Phase 3 Pass Gate

First run one example. Then run all ten only if the first example passes.

The example passes when:

- Aggregate coarse depth and mask metrics improve by at least 10%.
- All isometric, top, left, right, front, and back fresh-anchor evaluations are
  present.
- Worst-view P95 normalized depth error is at most `0.20`.
- Worst-view median normalized depth error is at most `0.15`.
- Worst-view soft mask IoU is at least `0.80` and does not regress.
- The result does not become transparent or empty.
- A normal decoder call with newly sampled anchors retains the improvement.
- The rich input's broad appearance remains recognizable.
- The improvement is visible in a primitive-wireframe overlay, not only in one
  scalar metric.

Save each accepted optimized latent as `target_latent.safetensors`. Preserve the
base camera token as `target_camera.safetensors` for the initial POC.

## Phase 4: Phase 2 LoRA Stress Test

This phase tests whether Phase 2 can learn the accepted
image-to-improved-latent mappings. It is intentionally allowed to memorize the
small clean subset.

### Trainable and Frozen Components

Frozen:

- DINOv3 image encoder.
- FLUX.2 VAE image encoder.
- Background removal model.
- Entire octree/Gaussian Phase 1 decoder.
- Original Phase 2 parameters.

Trainable:

- Rank-8 LoRA adapters on selected Phase 2 attention and MLP linear layers.

The repository does not use `transformers` or `diffusers`, so a small local
`LoRALinear` wrapper is preferable to importing a large framework only for the
POC.

### Flow-Matching Objective

For each optimized latent target `z_target`:

```text
epsilon ~ Normal(0, I)
t ~ Uniform(0, 1)
x_t = (1 - t) * z_target + t * epsilon
v_target = epsilon - z_target
v_pred = flow_model(x_t, 1000 * t, image_condition)
L_latent_flow = mean((v_pred - v_target)^2)
```

Apply the same interpolation and velocity objective to the preserved 5D camera
target, or freeze the camera-specific LoRA path and retain base camera behavior.
Do not silently omit the camera stream while passing random camera noise.

Use batch size 1, gradient accumulation, mixed precision, activation
checkpointing, and frequent checkpoints. The free 48 GB RTX A6000 should be a
reasonable target for this LoRA-scale test after conditioning features are
precomputed.

### Normal-Generation Evaluation

For every training image:

1. Run base TripoSplat from several fresh noise seeds.
2. Run the LoRA model from the same fresh seeds.
3. Decode normally with newly sampled octree anchors.
4. Render both from all six fixed scoring cameras.
5. Compare both against the exact primitive depth and mask.

Report:

```text
E_base   = baseline spatial error
E_target = optimized-latent spatial error
E_lora   = LoRA normal-generation spatial error

recovered_improvement =
    (E_base - E_lora) / max(E_base - E_target, epsilon)
```

The stress test passes when:

- Optimized targets are clearly better than baseline.
- LoRA outputs recover at least 50% of the target improvement on average.
- Improvement survives fresh flow noise and fresh octree sampling.
- No opacity, scale, or Gaussian-count collapse is observed.
- The GPT-generated broad appearance remains recognizable.

This result demonstrates memorization/trainability, not generalization.

## Supported Training Methods

### 1. Latent Optimization / Inversion

Use downstream spatial loss to optimize latent tokens while all model weights
are frozen. This is the required first method because it answers whether the
frozen decoder's latent space can express the desired correction.

Advantages:

- Small number of trainable values.
- Fast failure signal.
- Produces pseudo-target latents for conventional Phase 2 training.

Limitations:

- Per-example optimization cost.
- Non-unique latent targets.
- Hard octree sampling still requires the piecewise strategy above.

### 2. Supervised Flow-Matching LoRA

Train small Phase 2 adapters against accepted optimized latents. This is the
recommended POC stress test because it restores a standard local denoising
target instead of relying only on a terminal reward.

Advantages:

- Stable, ordinary flow-matching objective.
- Phase 1 remains untouched.
- Small checkpoint and manageable memory requirements.

Limitations:

- Quality is bounded by pseudo-target quality.
- Ten examples only test memorization.

### 3. Direct End-to-End Spatial Fine-Tuning

Backpropagate spatial loss through a differentiable multi-step Phase 2 sampler
and frozen decoder into LoRA parameters.

This is possible in principle but is not the first POC method. It retains a
large sampling graph, is expensive, and compounds the hard-octree problem with
terminal-reward instability.

### 4. Octree Structural Supervision

Construct level-by-level primitive surface histograms and apply cross-entropy to
the frozen octree density decoder while differentiating into latent tokens. This
matches the structural loss described in the paper and is the preferred
escalation if fixed-anchor depth optimization cannot move global density.

### 5. Future Spatial Control Adapter

After the reward and LoRA tests pass, GLB/primitive geometry can be sampled at
the 8192 fixed Sobol positions and embedded as one-to-one control features for
the latent tokens. That is a future ControlNet-like project and is outside this
POC.

## Proposed Implementation Layout

```text
training/
    scene_schema.py             primitive and camera artifact schema
    generate_poc_data.py        ten-scene artifact generation
    gaussian_depth_renderer.py  CPU evaluation depth/alpha renderer
    gsplat_depth_renderer.py    differentiable CUDA depth/alpha renderer
    spatial_loss.py             coarse tolerant spatial objective
    decode_train.py             differentiable fixed-anchor decoder path
    optimize_one_latent.py      one-scene Phase 3 latent inversion
    validate_gsplat_renderer.py CPU/CUDA renderer parity gate
    lora.py                     local LoRA linear wrapper
    train_flow_lora.py          Phase 4 flow-matching trainer
    evaluate.py                 baseline/target/LoRA comparison

configs/
    spatial_depth_poc.yaml

tests/
    test_spatial_loss.py
    test_decode_gradients.py
    test_flow_objective.py
```

Do not modify the public inference behavior in place. Add training-specific
entry points that call the existing components, and keep the original
`TripoSplatPipeline.run` path stable.

## Runtime Requirements

Recommended POC machine:

- Linux.
- NVIDIA RTX A6000 48 GB or better.
- Recent CUDA-compatible PyTorch.
- 64 GB or more system RAM.
- 20 GB persistent storage is sufficient for the one-scene test. Use at least
  50 GB before scaling to all examples and LoRA checkpoints.

GPU dependencies are listed in `requirements-gpu.txt`; the validated
differentiable rasterizer is `gsplat==1.5.3`.

## Minimal Execution Checklist

- [x] Reproduce base inference through the hosted TripoSplat service.
- [x] Generate all ten primitive/GPT-image artifact bundles.
- [x] Calibrate the fixed isometric scoring camera.
- [x] Add deterministic top and side scoring cameras.
- [x] Add deterministic left, right, front, and back scoring cameras.
- [x] Render baseline splat depth and alpha.
- [x] Verify spatial score perturbation tests.
- [x] Verify nonzero finite gradients from depth loss to latent tokens.
- [x] Optimize one latent and verify a fresh normal decode improves.
- [x] Generate and optimize all ten examples.
- [x] Add rank-8 Phase 2 LoRA.
- [x] Overfit the ten pseudo-target latents with flow matching.
- [x] Reject stale one-view and high-P95 pseudo-targets.
- [x] Overfit the clean multi-view pseudo-target subset with flow matching.
- [x] Re-optimize and retrain with six-view supervision.
- [x] Evaluate fresh-noise LoRA outputs against base and target.
- [x] Record pass/fail decision before increasing dataset size or architecture.
- [x] Evaluate three entirely held-out generated-image/primitive layouts.
- [x] Correct the held-out alpha/preprocessing confound and rerun all pairs.
- [x] Record early positive but still limited held-out generalization.
- [x] Scaffold 50 diverse grounded and floorless primitive contracts.

## Decision After the POC

If Phase 3 fails, the next work is decoder reachability and octree structural
supervision, not more Phase 2 training.

If Phase 3 passes but Phase 4 fails, debug target consistency, LoRA placement,
flow/camera objectives, and sampling before adding more data.

If both pass, scale in this order:

1. Add held-out primitive arrangements.
2. Add randomized oblique camera sampling or direct 3D supervision.
3. Add exact SDF and octree surface supervision.
4. Add varied generated-image styles.
5. Evaluate a direct 3D spatial-control adapter.

## Diverse Final-LoRA Run (2026-07-18)

This is the active end-to-end run over the 50 diverse ControlNet-generated
conditions in `poc_data/diverse_train`.

### Frozen split

`training.create_data_split` writes `poc_data/diverse_train/split.json` once.
Each of the ten semantic categories contributes four training scenes. Its fifth
scene alternates between validation and test, producing a disjoint 40/5/5
split. The test five are not inspected by candidate selection, latent
optimization, target filtering, checkpoint selection, or hyperparameter
selection.

### Structure-first evaluator

Primitive IDs divide each target render into:

- object geometry: every non-ground primitive;
- support geometry: primitives explicitly named `ground`, `floor`, or
  `terrain`;
- the whole scene as a low-weight guardrail.

The differentiable latent objective weights object, support, and whole-scene
losses `1.0`, `0.5`, and `0.15`. The evaluator reports object P95 depth,
object mask IoU, object bounding-box IoU, centroid, extent, signal ratio,
support P95 depth, support coverage, and support flatness. Six directional
views are mandatory.

The default hard target gate requires object P95 depth at most `0.20`, object
soft IoU at least `0.65`, bounding-box IoU at least `0.49`, centroid error at
most `0.10`, extent error at most `0.12`, and signal ratio at least `0.70`.
Grounded scenes additionally require support P95 at most `0.20`, support
flatness error at most `0.08`, and support coverage at least `0.70`. An
optimized target must retain at least 90% of its base object signal.

Synthetic tests prove that a large correct floor cannot hide flipped or
shifted objects, and that weak-signal objects and warped floors fail. The same
gate also separates the known good and bad examples from the original
ten-scene experiment.

### Candidate and target generation

For every train/validation condition, Phase 2 samples seeds 41-44. Each sample
is decoded with the same octree seed and scored in six views. Gross
orientation, bounding box, centroid, extent, and signal checks rank ahead of
raw loss. Failed scenes receive one second four-seed bank; a final failure is
excluded.

The selected base latent is then optimized against the object/support-weighted
six-view objective while Phase 1 remains frozen. A fresh octree sample, not the
optimization anchors, determines target acceptance. Failed targets receive one
longer retry and are then excluded. The desired minimum is 30 balanced accepted
training targets for a later production-scale pass. This POC was allowed to
continue with fewer targets so that the full generalization test could be run
before paying to generate and optimize another dataset.

### LoRA selection and final test

Two rank-8 LoRA candidates use the accepted training split with learning rates
`1e-4` and `5e-5`. Paired fresh generations on the five validation scenes
select the lower structure-loss candidate, with P95 depth as the primary
diagnostic. The selected settings are retrained from the base model on accepted
train plus validation targets.

The final adapter is evaluated once on the five untouched test conditions with
paired seeds 101, 202, and 303. The package includes the adapter, config,
training history, split, accepted manifests, base/LoRA splats, six-view metrics,
and viewer-compatible artifacts.

### Executed result

The run completed end to end on the A40 RunPod instance:

- 50 diverse primitive contracts were split into 40 train, 5 validation, and 5
  untouched test scenes.
- Candidate generation found a viable starting orientation for 42 of the 45
  train/validation scenes. `32_pipe_cluster`, `39_flying_drone`, and
  `47_orbit_sculpture` were rejected before latent optimization.
- Six-view latent optimization produced 15 accepted training targets and 3
  accepted validation targets. The final adapter therefore trained on 18
  pseudo-targets.
- Two rank-8, alpha-8 candidates trained for 600 steps. On 10 paired validation
  generations, learning rate `1e-4` improved mean structure loss by 15.56% and
  learning rate `5e-5` improved it by 14.27%. The `5e-5` candidate won P95 depth
  on 7/10 pairs versus 6/10, so it was selected because tail depth was the
  stated priority.
- The selected settings were retrained from the untouched base for 800 steps on
  all 18 accepted train/validation targets. The fixed training probe improved
  from `0.6298` to `0.5270` structure loss, a 16.33% reduction.

The one-time final test used Palm Oasis, Igloo, Garden Bench, Cylinder Droid,
and Asymmetric Sculpture with paired seeds 101, 202, and 303. Across all 15
pairs, mean structure loss fell from `0.6572` to `0.5463`, a mean relative
improvement of 14.36%. The LoRA won 13/15 structure comparisons, and every test
scene improved when averaged over its three seeds.

P95 depth was less conclusive: its mean was effectively flat (`1.2676` base,
`1.2671` LoRA), although the LoRA won 10/15 individual P95 comparisons. This is
a positive proof that Phase 2 LoRA training can transfer the optimized spatial
targets to unseen conditions, but it is not yet a solved tail-geometry model.
The next run should increase accepted-target diversity and put more direct
weight on worst-view P95 failures before increasing LoRA capacity.

Final artifacts:

- adapter: `poc_data/diverse_lora_final/flow_lora.safetensors`
- training record: `poc_data/diverse_lora_final/train_summary.json`
- frozen split and accepted sets: `poc_data/diverse_train/split.json` and
  `poc_data/diverse_train/lora_*_manifest.json`
- untouched test metrics and splats: `poc_data/diverse_test_final`
- visual comparison: `poc_viewer.html`, under the `Final untouched test` scene
  group with `Fresh base` and `Fresh LoRA`

## Primitive Geometry Adapter Run (2026-07-19)

The first ControlNet-style Phase 2 adapter is implemented in
`spatial_control.py`. It converts each primitive contract into 12 analytic SDF
features at TripoSplat's 8,192 fixed Sobol token positions, then injects
zero-initialized residuals into six frozen flow-transformer blocks at every
sampling step. Phase 1 remains unchanged. The adapter has 1,209,600 trainable
parameters and can be composed with the existing rank-8 LoRA.

The strict six-view target gate accepted seven training scenes and two held-out
validation scenes. The adapter trained for 600 steps at `1e-4`, with LoRA
enabled on half of the steps so both deployment combinations remained usable.
Training took 1,016 seconds on an A40 and peaked at 2.72 GiB VRAM.

The base held-out latent-flow probe fell from `0.6827` to `0.5907`. The LoRA
probe was flat (`0.5769` to `0.5770`), and a shuffled control scored `0.5908`,
which warns that this small adapter still contains a substantial generic
correction rather than proven scene-specific control.

The paired fresh-generation test used seed 101 on held-out Pillars and Crate
Yard. Every mode received the same encoded image and exact Phase 2 noise.
Geometry Control beat Base on both scenes: mean six-view structure loss fell
from `0.1051` to `0.0942`, mean worst-view P95 depth fell from `0.4519` to
`0.3971`, and mean minimum soft IoU rose from `0.8629` to `0.8758`. Combined
LoRA plus control improved structure loss to `0.0988`, but won only one of the
two scene pairs. More diverse accepted targets and shuffled-control
regularization are the next training priorities.

Final artifacts:

- adapter and training record: `poc_data/spatial_control_run`
- paired four-way splats and six-view metrics:
  `poc_data/control_generation_eval`
- train and validation manifests: `poc_data/control_train_manifest.json` and
  `poc_data/control_validation_manifest.json`
- hosted comparison: `https://huggingface.co/spaces/WilliamQM/Spatial-Splat`

## Ultra-Low-Rank Probe (2026-07-19)

A fixed output-latent basis was tested first using the four accepted original
six-view target corrections. Although rank four reconstructs those four
training corrections exactly, it retained usually less than 1% of the fresh
LoRA correction energy. Final latent coordinates therefore did not behave like
a reusable set of global transform controls in this small probe.

The deployable alternative compresses every trained rank-8 LoRA layer with a
truncated SVD. Effective rank two retains 87.60% of the total learned
matrix-update energy while reducing the adapter from 3,670,016 to 917,504
parameters. This is a compression experiment, not evidence that the retained
directions exclusively control position, rotation, and scale.

Artifacts:

- compressor: `training/compress_lora.py`
- rank-2 adapter: `poc_data/diverse_lora_rank2`
- five-way hosted test: Base, rank-8 LoRA, rank-2 LoRA, Geometry Control, and
  rank-8 LoRA plus Geometry Control

The first hosted five-way smoke test used Cylinder Droid, seed 42, 20 flow
steps, and 32,768 Gaussians. Rank two closely reproduced rank eight but did not
improve it: structure loss was `0.7901` versus `0.7834`, and worst-view object
P95 was `1.0925` versus `1.0779`. Base scored `0.6384`; Geometry Control was
best at `0.5901`. Simple weight compression therefore preserves much of the
LoRA behavior but does not isolate a quality-preserving transform mechanism.
The complete record is `poc_data/rank2_space_smoke/spatial_metrics.json`.

## Scheduled Geometry Control Probe (2026-07-19)

Geometry control can now stop before the flow sampler finishes. The hosted
default applies control for the first 70% of steps and passes `control=None`
for the final 30%, giving the unmodified Phase 2 model six cleanup steps in a
20-step run. The Space exposes both cutoff (`0` to `1`) and control strength
(`0` to `15`); cutoff `1` reproduces the original always-on path.

On Cylinder Droid at seed 42 and control strength 1, scheduled control changed
Geometry Control structure loss from `0.5901` to `0.5958` and worst-view P95
from `0.8380` to `0.8354`. Combined LoRA plus Control was essentially flat in
structure loss (`0.6024` to `0.6023`) while P95 worsened from `0.9079` to
`0.9220`. This proves the schedule works but does not show a quality gain at
the trained strength. Strong-early-control sweeps are the next useful test.
The scheduled result is in `poc_data/early_control_space_smoke`.

## Procedural Six-View LoRA Probe (2026-07-20)

This experiment tests whether exact primitive renders can scale spatial target
generation without relying on one generated-image camera. The deterministic
dataset in `poc_data/procedural_multiview` contains 20 scenes across ten shape
families. Each scene provides one isometric RGBA Phase 2 input plus exact RGB,
depth, mask, boundary, and primitive-ID supervision from isometric, top, left,
right, front, and back cameras. The frozen split is 12 train, 4 validation, and
4 test scenes.

Latent optimization now combines six-view structure loss with premultiplied
RGB anchoring and fixed-anchor feature, opacity, scale, and effective-density
preservation. A weak density penalty allowed one robot target to shrink its
effective Gaussian mass while improving geometry, so the final run uses
weight `5.0` and a `0.98` minimum-density ratio. The stricter robot target kept
the RGB match, retained density, and still improved fresh structure by 20.9%.
The procedural manifest can explicitly skip the older realistic-data support
gate while retaining six-view depth, IoU, visual, opacity, and density checks.

The first fresh rank-4 LoRA trained for 400 steps on 10 accepted train targets.
Its fixed flow probe improved 9.0%, but the untouched four-scene test was
negative: two wins, median structure change -6.6%, and mean change -35.8% due
to a large cluster regression. No accepted cluster or stairs target existed in
that training set.

A second fresh rank-4 LoRA trained for 300 steps on 12 accepted
train/validation targets, adding quality-gated cluster and stairs examples
while continuing to exclude all four test scenes. Its fixed flow probe
improved 7.6%. On the same paired seed-101 test it won 3/4 scenes: stairs
improved 15.9%, table 4.4%, and tower 4.7%. Cluster still regressed 48.8%, so
median improvement was +4.5% but mean improvement remained -6.0%. Mean P95
depth was also slightly worse (`0.3057` to `0.3118`).

This is useful evidence that exact six-view procedural targets can teach
transferable corrections, but 12 accepted examples are not enough for robust
family generalization. Do not replace the existing diverse LoRA with this
adapter. Scale scene-family coverage, use multiple held-out seeds, and require
both mean structure and P95 wins before promoting a future version.

Artifacts:

- generator: `training/generate_procedural_multiview_data.py`
- visual and density losses: `training/visual_anchor_loss.py`
- final adapter: `poc_data/procedural_multiview_lora_v2`
- first and second paired tests: `poc_data/procedural_lora_test_eval` and
  `poc_data/procedural_lora_v2_test_eval`
- hosted experimental mode:
  `https://huggingface.co/spaces/WilliamQM/Spatial-Splat`
