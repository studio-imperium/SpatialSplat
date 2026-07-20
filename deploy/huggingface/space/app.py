"""Private ZeroGPU comparison harness for Spatial Splat adapters."""

import json
from pathlib import Path
import threading
import time
from uuid import uuid4

import gradio as gr
from huggingface_hub import snapshot_download
import spaces
import torch

from space_metrics import build_score_targets, metrics_html, score_gaussian
from spatial_control import (
    canonicalize_scene,
    load_spatial_control,
    scene_control_tensor,
)
from spatial_lora import (
    load_spatial_lora,
    load_spatial_lora_weights,
    set_lora_enabled,
)
from triposplat import TripoSplatPipeline


snapshot_download(repo_id="VAST-AI/TripoSplat", local_dir="ckpts")

PIPE = TripoSplatPipeline(
    ckpt_path="ckpts/diffusion_models/triposplat_fp16.safetensors",
    decoder_path="ckpts/vae/triposplat_vae_decoder_fp16.safetensors",
    dinov3_path="ckpts/clip_vision/dino_v3_vit_h.safetensors",
    flux2_vae_encoder_path="ckpts/vae/flux2-vae.safetensors",
    rmbg_path="ckpts/background_removal/birefnet.safetensors",
    device="cuda",
)
load_spatial_lora(
    PIPE.flow_model,
    "adapter/flow_lora.safetensors",
    "adapter/flow_lora_config.json",
)
FULL_LORA = (
    Path("adapter/flow_lora.safetensors"),
    Path("adapter/flow_lora_config.json"),
)
LOW_RANK_LORA = (
    Path("adapter/flow_lora_rank2.safetensors"),
    Path("adapter/flow_lora_rank2_config.json"),
)
CONTROL_CONFIG = load_spatial_control(
    PIPE.flow_model,
    "adapter/spatial_control.safetensors",
    "adapter/spatial_control_config.json",
    device="cuda",
    dtype=torch.float16,
)
PIPE.flow_model.eval()

OUT_ROOT = Path("gradio_outputs").resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)
VIEWER_HTML = Path("static/viewer/viewer.html").resolve()
TEMPLATE_ROOT = Path("examples").resolve()
PIPE_LOCK = threading.Lock()
MODES = (
    ("base", "Base TripoSplat", None, False),
    ("lora", "Spatial LoRA", FULL_LORA, False),
    ("low_rank", "Rank-2 Spatial LoRA", LOW_RANK_LORA, False),
    ("control", "Geometry Control", None, True),
    ("combined", "LoRA + Geometry Control", FULL_LORA, True),
)
TEST_TEMPLATE_GROUPS = (
    (
        "New realistic test templates",
        (
            ("pine_clearing", "Pine Clearing"),
            ("cactus_desert", "Cactus Desert"),
            ("wizard_tower", "Wizard Tower"),
        ),
    ),
    (
        "Earlier synthetic tests",
        (
            ("palm_oasis", "Palm Oasis"),
            ("garden_bench", "Garden Bench"),
            ("cylinder_droid", "Cylinder Droid"),
        ),
    ),
)
TEST_TEMPLATES = tuple(
    template
    for _, templates in TEST_TEMPLATE_GROUPS
    for template in templates
)


def _gradio_file(path: Path) -> str:
    return f"/gradio_api/file={path.as_posix()}"


def _viewer_iframe(ply_path: Path, scene_path: Path) -> str:
    src = (
        f"{_gradio_file(VIEWER_HTML)}?ply={_gradio_file(ply_path)}"
        f"&scene={_gradio_file(scene_path)}&ts={time.time()}"
    )
    return (
        f"<iframe src='{src}' "
        "style='width:100%;height:460px;border:1px solid #999;background:#fff'></iframe>"
    )


def _read_scene(scene_file) -> dict:
    if scene_file is None:
        raise gr.Error("Upload the primitive scene.json used to create the image.")
    path = Path(scene_file)
    try:
        scene = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise gr.Error(f"Could not read scene JSON: {error}") from error
    try:
        return canonicalize_scene(scene)
    except (TypeError, ValueError) as error:
        raise gr.Error(f"Unsupported primitive JSON: {error}") from error


def _load_test_template(slug: str) -> tuple[str, str]:
    valid_slugs = {template_slug for template_slug, _ in TEST_TEMPLATES}
    if slug not in valid_slugs:
        raise gr.Error("Unknown test template.")
    template_dir = TEMPLATE_ROOT / slug
    return (
        str(template_dir / "generated_image.png"),
        str(template_dir / "scene.json"),
    )


@spaces.GPU(duration=75)
def generate_comparison(
    image,
    scene_file,
    seed: int,
    steps: int,
    guidance_scale: float,
    control_scale: float,
    num_gaussians: str,
    output_format: str,
    progress=gr.Progress(track_tqdm=True),
):
    if image is None:
        raise gr.Error("Upload an image first.")
    scene = _read_scene(scene_file)
    out_dir = OUT_ROOT / uuid4().hex[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_path = out_dir / "scene.json"
    scene_path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")

    viewers = []
    downloads = []
    result_lines = []
    metric_results = {}
    started = time.time()
    with PIPE_LOCK:
        progress(0, desc="Preprocessing and encoding")
        prepared = PIPE.preprocess_image(image)
        condition_generator = torch.Generator(device=PIPE._device).manual_seed(
            int(seed)
        )
        conditioning = PIPE.encode_image(prepared, generator=condition_generator)
        sample_rng_state = condition_generator.get_state()
        control = scene_control_tensor(
            scene,
            token_count=PIPE.flow_model.q_token_length,
            device=PIPE._device,
        )
        progress(0.05, desc="Building six-view spatial targets")
        score_targets = build_score_targets(scene)

        for index, (slug, label, lora_variant, use_control) in enumerate(MODES):
            progress(index / len(MODES), desc=f"Sampling {label}")
            if lora_variant is None:
                set_lora_enabled(PIPE.flow_model, False)
            else:
                load_spatial_lora_weights(
                    PIPE.flow_model, lora_variant[0], lora_variant[1]
                )
                set_lora_enabled(PIPE.flow_model, True)
            sample_generator = torch.Generator(device=PIPE._device)
            sample_generator.set_state(sample_rng_state)
            mode_started = time.time()
            output = PIPE.sample_latent(
                conditioning,
                steps=int(steps),
                guidance_scale=float(guidance_scale),
                generator=sample_generator,
                show_progress=True,
                control=control if use_control else None,
                control_scale=float(control_scale),
            )
            progress((index + 0.85) / len(MODES), desc=f"Decoding {label}")
            gaussian = PIPE.decode_latent(
                output["latent"], num_gaussians=int(num_gaussians)
            )
            ply_path = out_dir / f"{slug}.ply"
            gaussian.save_ply(str(ply_path))
            if output_format == "splat":
                download_path = out_dir / f"{slug}.splat"
                gaussian.save_splat(str(download_path))
            else:
                download_path = ply_path
            progress((index + 0.9) / len(MODES), desc=f"Scoring {label} in six views")
            metric_results[slug] = score_gaussian(gaussian, score_targets)
            elapsed = time.time() - mode_started
            viewers.append(_viewer_iframe(ply_path, scene_path))
            downloads.append(str(download_path))
            result_lines.append(
                f"{label}: {gaussian.get_xyz.shape[0]:,} gaussians, {elapsed:.1f}s"
            )
            del output, gaussian

        set_lora_enabled(PIPE.flow_model, False)

    progress(1, desc="Done")
    metrics_path = out_dir / "spatial_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "render_size": score_targets["render_size"],
                "views": ["isometric", "top", "left", "right", "front", "back"],
                "modes": metric_results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    info = "\n".join(result_lines)
    info += f"\nTotal: {time.time() - started:.1f}s"
    return prepared, *viewers, *downloads, metrics_html(metric_results), str(metrics_path), info


with gr.Blocks(title="Spatial Splat") as demo:
    gr.Markdown("# Spatial Splat comparison")
    with gr.Row():
        image_input = gr.Image(
            label="Generated image", type="pil", image_mode="RGBA", height=280
        )
        scene_input = gr.File(
            label="Matching primitive scene.json",
            file_types=[".json"],
            type="filepath",
        )
        prepared_output = gr.Image(label="Preprocessed input", height=280)

    template_buttons = []
    for group_label, templates in TEST_TEMPLATE_GROUPS:
        gr.Markdown(f"**{group_label}**")
        with gr.Row():
            for slug, label in templates:
                template_buttons.append((gr.Button(label), slug))

    with gr.Row():
        seed_input = gr.Number(label="Seed", value=42, precision=0)
        steps_input = gr.Slider(1, 30, value=20, step=1, label="Steps")
        guidance_input = gr.Slider(
            1.0, 7.0, value=3.0, step=0.5, label="Image guidance"
        )
        control_input = gr.Slider(
            0.0,
            2.0,
            value=float(CONTROL_CONFIG.get("default_control_scale", 1.0)),
            step=0.1,
            label="Geometry control",
        )
        gaussians_input = gr.Dropdown(
            ["32768", "65536", "131072"], value="32768", label="Gaussians"
        )
        format_input = gr.Radio(["ply", "splat"], value="ply", label="Download")
        generate_button = gr.Button("Run five-way comparison", variant="primary")

    viewer_outputs = []
    file_outputs = []
    for row_modes in (MODES[:3], MODES[3:]):
        with gr.Row():
            for slug, label, _, _ in row_modes:
                with gr.Column():
                    gr.Markdown(f"### {label}")
                    viewer_outputs.append(gr.HTML())
                    file_outputs.append(gr.File(label=f"Download {label}"))
    gr.Markdown("## Six-view spatial metrics")
    metrics_output = gr.HTML()
    metrics_file_output = gr.File(label="Download spatial metrics JSON")
    info_output = gr.Textbox(label="Run summary", interactive=False, lines=6)

    for template_button, template_slug in template_buttons:
        template_button.click(
            lambda slug=template_slug: _load_test_template(slug),
            inputs=None,
            outputs=[image_input, scene_input],
            queue=False,
        )

    generate_button.click(
        generate_comparison,
        inputs=[
            image_input,
            scene_input,
            seed_input,
            steps_input,
            guidance_input,
            control_input,
            gaussians_input,
            format_input,
        ],
        outputs=[
            prepared_output,
            *viewer_outputs,
            *file_outputs,
            metrics_output,
            metrics_file_output,
            info_output,
        ],
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(
        allowed_paths=[str(VIEWER_HTML.parent), str(TEMPLATE_ROOT), str(OUT_ROOT)],
        show_error=True,
    )
