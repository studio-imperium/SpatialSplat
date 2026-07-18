# SpatialSplat Isometric Depth POC Plan

## Status

This document describes the first intentionally small proof of concept for
spatially supervised TripoSplat training. It is also an engineering handoff for
agents working in this repository.

The POC asks one question:

> Can a rich image generated from a primitive layout be mapped by TripoSplat to
> a Gaussian splat whose coarse visible geometry agrees with the original
> primitives from one fixed isometric viewpoint?

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
  latent-only optimization, and fixed/fresh-anchor evaluation.
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
  and finite gradients to predicted depth and alpha.

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

Pending POC work:

- Run normal generation from fresh flow-noise seeds through base and selected
  LoRA checkpoints, then decode with fresh octree anchors and score spatially.
- Record the Phase 4 pass/fail decision before increasing data or architecture.

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
  --optimization-steps 60 --render-size 256 --final-render-size 512
.venv/bin/python -m training.optimize_latent_batch \
  --data-root poc_data --checkpoint-root ckpts
.venv/bin/python -m training.build_lora_dataset \
  --data-root poc_data --output poc_data/lora_dataset.json
.venv/bin/python -m training.train_flow_lora \
  --manifest poc_data/lora_dataset.json --checkpoint-root ckpts \
  --output-dir poc_data/lora_run --steps 1000
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
    -> depth and alpha from the same isometric camera
    -> coarse spatial score against exact primitive depth and mask
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
- A one-view depth score cannot supervise hidden geometry. This POC evaluates
  visible isometric alignment only.
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
- Multiple supervision cameras.
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
- Fixed scoring camera.
- Anchor resampling every 10 to 25 steps.
- Save the lowest-loss latent, not merely the final latent.

### Phase 3 Pass Gate

First run one example. Then run all ten only if the first example passes.

The example passes when:

- Coarse depth and mask metrics improve materially.
- The result does not become transparent or empty.
- A normal decoder call with newly sampled anchors retains the improvement.
- The rich input's broad appearance remains recognizable.
- The improvement is visible in a primitive-wireframe overlay, not only in one
  scalar metric.

Save each accepted optimized latent as `target_latent.safetensors`. Preserve the
base camera token as `target_camera.safetensors` for the initial POC.

## Phase 4: Phase 2 LoRA Stress Test

This phase tests whether Phase 2 can learn the ten image-to-improved-latent
mappings. It is intentionally allowed to memorize all ten examples.

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
4. Render both from the fixed isometric scoring camera.
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
- [x] Render baseline splat depth and alpha.
- [x] Verify spatial score perturbation tests.
- [x] Verify nonzero finite gradients from depth loss to latent tokens.
- [x] Optimize one latent and verify a fresh normal decode improves.
- [x] Generate and optimize all ten examples.
- [x] Add rank-8 Phase 2 LoRA.
- [x] Overfit the ten pseudo-target latents with flow matching.
- [ ] Evaluate fresh-noise LoRA outputs against base and target.
- [ ] Record pass/fail decision before increasing dataset size or architecture.

## Decision After the POC

If Phase 3 fails, the next work is decoder reachability and octree structural
supervision, not more Phase 2 training.

If Phase 3 passes but Phase 4 fails, debug target consistency, LoRA placement,
flow/camera objectives, and sampling before adding more data.

If both pass, scale in this order:

1. Add held-out primitive arrangements.
2. Add multiple scoring cameras.
3. Add exact SDF and octree surface supervision.
4. Add varied generated-image styles.
5. Evaluate a direct 3D spatial-control adapter.
