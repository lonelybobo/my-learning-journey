"""Command-line demo for CtrlRegen / CtrlRegen+ watermark removal.

The method regenerates an image under two conditions extracted from the
watermarked input:

  * semantic condition: DINOv2-giant features -> Resampler -> IP cross-attention
  * spatial  condition: Canny edges              -> ControlNet

Two inference modes are supported:

  * ``ctrlregen``      : start from pure Gaussian noise and run the reverse
                         diffusion loop only (no VAE encoding of the input).
  * ``ctrlregen-plus`` : start from a partially noised input latent; ``step``
                         in [0, 1] controls the watermark/fidelity trade-off
                         (0 returns the resized input).

Examples:
    # CtrlRegen from pure noise (default, no --step)
    python demo.py --input examples/1.jpg --seed 1

    # CtrlRegen+ with the paper's example strength
    python demo.py --input examples/1.jpg --step 0.5 --seed 1

    # Batch processing
    python demo.py --input examples --output-dir outputs --step 0.5
"""

import argparse
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from controlnet_aux import CannyDetector
from diffusers import AutoencoderKL, ControlNetModel, UniPCMultistepScheduler
from transformers import AutoImageProcessor, AutoModel

from custom_i2i_pipeline import CustomStableDiffusionControlNetImg2ImgPipeline
from utils import color_match

ROOT = Path(__file__).resolve().parent

# Model locations. Override these paths if your checkpoints live elsewhere
# (see README.md / CODE_WALKTHROUGH.md).
DIFFUSION_MODEL = "SG161222/Realistic_Vision_V4.0_noVAE"
SPATIAL_CONTROL_PATH = ROOT / "models" / "spatialnet_ckp" / "spatial_control_ckp_14000"
SEMANTIC_CONTROL_PATH = ROOT / "models" / "semanticnet_ckp"
SEMANTIC_CONTROL_NAME = "semantic_control_ckp_435000.bin"
IMAGE_ENCODER = "facebook/dinov2-giant"
VAE = "stabilityai/sd-vae-ft-mse"

PROMPT = "best quality, high quality"
NEGATIVE_PROMPT = "monochrome, lowres, bad anatomy, worst quality, low quality"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

MODE_CTRLREGEN = "ctrlregen"
MODE_CTRLREGEN_PLUS = "ctrlregen-plus"
MODES = (MODE_CTRLREGEN, MODE_CTRLREGEN_PLUS)

transform_size_to_512 = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
    ]
)

# Models are loaded lazily so that `python demo.py --help` does not download or
# allocate anything. They are reused for every image in a batch.
_PIPE = None
_CANNY = None
_PIPE_DEVICE = None


def _resolve_device(device=None):
    """Normalize a CLI device string to ``torch.device``."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        device = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid device: {device!r}") from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device.index} is out of range "
                f"(found {torch.cuda.device_count()} device(s))."
            )
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
    return device


def load_models(device=None):
    """Build and cache the CtrlRegen pipeline.

    Returns:
        (pipeline, canny_detector)
    """
    global _PIPE, _CANNY, _PIPE_DEVICE

    target_device = _resolve_device(device)
    if _PIPE is not None:
        if _PIPE_DEVICE != target_device:
            raise RuntimeError(
                f"Models are already loaded on {_PIPE_DEVICE}; "
                f"cannot move them to {target_device} in this process."
            )
        return _PIPE, _CANNY

    dtype = torch.float16 if target_device.type == "cuda" else torch.float32

    # --- spatial control network: a ControlNet trained on Canny edges ---
    spatialnet = [
        ControlNetModel.from_pretrained(str(SPATIAL_CONTROL_PATH), torch_dtype=dtype)
    ]
    pipe = CustomStableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        DIFFUSION_MODEL,
        controlnet=spatialnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # --- semantic control adapter: Resampler + decoupled IP cross-attention ---
    # `costum_load_ip_adapter` (sic) is this repo's DINOv2-compatible loader.
    # image_encoder_folder=None skips diffusers' default CLIP loader; the DINOv2
    # encoder and processor are installed explicitly below.
    pipe.costum_load_ip_adapter(
        str(SEMANTIC_CONTROL_PATH),
        subfolder="models",
        weight_name=SEMANTIC_CONTROL_NAME,
        image_encoder_folder=None,
    )
    pipe.image_encoder = AutoModel.from_pretrained(IMAGE_ENCODER, torch_dtype=dtype)
    pipe.feature_extractor = AutoImageProcessor.from_pretrained(IMAGE_ENCODER)

    # The base checkpoint is the "noVAE" variant, so the VAE used for training
    # (sd-vae-ft-mse) is loaded explicitly.
    pipe.vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=dtype)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_ip_adapter_scale(1.0)
    pipe.set_progress_bar_config(disable=True)
    pipe.to(target_device)

    _PIPE = pipe
    _CANNY = CannyDetector()
    _PIPE_DEVICE = target_device
    return _PIPE, _CANNY


def _ensure_models(pipe, canny):
    """Resolve lazily loaded components when a public helper is called directly."""
    if pipe is None:
        loaded_pipe, loaded_canny = load_models()
        return loaded_pipe, canny if canny is not None else loaded_canny
    if canny is None:
        canny = CannyDetector()
    return pipe, canny


def _prepare_image(image):
    """Convert an arbitrary PIL image to the RGB 512x512 input format."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"expected a PIL image, got {type(image)!r}")
    image = ImageOps.exif_transpose(image)
    return transform_size_to_512(image.convert("RGB"))


def _make_control_image(canny, image, low, high):
    if not 0 <= low < high <= 255:
        raise ValueError(
            f"Canny thresholds must satisfy 0 <= low < high <= 255, got {low}, {high}"
        )
    return canny(image, low_threshold=low, high_threshold=high)


def _make_generator(pipe, seed):
    return torch.Generator(device=pipe.device).manual_seed(seed)


def _run_pipeline(
    pipe,
    input_img,
    control_img,
    *,
    num_inference_steps,
    guidance_scale,
    controlnet_scale,
    generator,
    latents=None,
    strength=None,
):
    """Call the underlying pipeline for either CtrlRegen or CtrlRegen+."""
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")
    if guidance_scale <= 0:
        raise ValueError("guidance_scale must be > 0")
    if controlnet_scale < 0:
        raise ValueError("controlnet_scale must be >= 0")

    kwargs = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        # When `latents` is provided (CtrlRegen), `image` is only a placeholder
        # and diffusers skips prepare_latents(), so the input is not VAE-encoded.
        "image": [input_img],
        "control_image": [control_img],
        "ip_adapter_image": [input_img],
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "controlnet_conditioning_scale": controlnet_scale,
        "control_guidance_start": 0.0,
        "control_guidance_end": 1.0,
        "generator": generator,
    }

    if latents is None:
        if strength is None:
            raise ValueError("strength is required when latents is not provided")
        kwargs["strength"] = strength
    else:
        # Supplying latents bypasses the img2img VAE encode/add_noise path.
        # strength must be 1.0 so the full reverse trajectory is used.
        kwargs["latents"] = latents
        kwargs["strength"] = 1.0

    return pipe(**kwargs).images[0]


def _finish(input_img, output_img, match_colors):
    return color_match(input_img, output_img) if match_colors else output_img


def ctrl_regen(
    input_img,
    seed=0,
    pipe=None,
    canny=None,
    num_inference_steps=50,
    guidance_scale=2.0,
    match_colors=True,
    controlnet_scale=1.0,
    canny_low=100,
    canny_high=150,
):
    """Pure CtrlRegen: start from Gaussian noise and run reverse diffusion only.

    Unlike ``ctrl_regen_plus``, this does not VAE-encode the input. The input is
    still used to extract the semantic (DINOv2) and spatial (Canny) conditions.
    """
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")

    pipe, canny = _ensure_models(pipe, canny)
    input_img = _prepare_image(input_img)
    control_img = _make_control_image(canny, input_img, canny_low, canny_high)

    generator = _make_generator(pipe, seed)
    vae_scale_factor = getattr(pipe, "vae_scale_factor", 8)
    height = input_img.height // vae_scale_factor
    width = input_img.width // vae_scale_factor
    latents = torch.randn(
        (1, pipe.unet.config.in_channels, height, width),
        generator=generator,
        device=pipe.device,
        dtype=pipe.unet.dtype,
    ) * float(pipe.scheduler.init_noise_sigma)

    output_img = _run_pipeline(
        pipe,
        input_img,
        control_img,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_scale=controlnet_scale,
        generator=generator,
        latents=latents,
    )
    return _finish(input_img, output_img, match_colors)


def ctrl_regen_plus(
    input_img,
    step,
    seed=0,
    pipe=None,
    canny=None,
    num_inference_steps=50,
    guidance_scale=2.0,
    match_colors=True,
    controlnet_scale=1.0,
    canny_low=100,
    canny_high=150,
):
    """CtrlRegen+: img2img regeneration with a partially noised input latent.

    ``step`` is diffusers' ``strength`` and must be in [0, 1]. Larger values
    remove more of the watermark and change the image more. ``step=1`` is close
    to CtrlRegen but still goes through the img2img ``prepare_latents`` path;
    use :func:`ctrl_regen` for the exact pure-noise start.
    """
    if not 0.0 <= step <= 1.0:
        raise ValueError(f"step must be in [0, 1], got {step}")
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be >= 1")

    pipe, canny = _ensure_models(pipe, canny)
    input_img = _prepare_image(input_img)

    # diffusers' img2img needs at least one denoising step; for a strength that
    # rounds to zero, just return the resized input.
    if step <= 0.0 or int(num_inference_steps * step) == 0:
        return input_img.copy()

    control_img = _make_control_image(canny, input_img, canny_low, canny_high)
    generator = _make_generator(pipe, seed)

    output_img = _run_pipeline(
        pipe,
        input_img,
        control_img,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_scale=controlnet_scale,
        generator=generator,
        strength=step,
    )
    return _finish(input_img, output_img, match_colors)


def collect_inputs(path):
    """Return image files from a file or a non-recursive directory."""
    path = Path(path)
    if path.is_dir():
        return sorted(
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"unsupported image extension: {path.suffix}")
    return [path]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Remove invisible image watermarks with CtrlRegen / CtrlRegen+."
    )

    io_group = parser.add_argument_group("input / output")
    io_group.add_argument(
        "--input",
        type=Path,
        default=ROOT / "examples" / "hide.png",
        help="Input image or a non-recursive directory of images.",
    )
    io_group.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
        help="Directory where regenerated images are written.",
    )

    regen_group = parser.add_argument_group("regeneration")
    regen_group.add_argument(
        "--mode",
        choices=MODES,
        default="ctrlregen",
        help=(
            "ctrlregen: pure-noise reverse diffusion (no --step needed); "
            "ctrlregen-plus: img2img with --step. "
            "Default: ctrlregen-plus when --step is given, otherwise ctrlregen."
        ),
    )
    regen_group.add_argument(
        "--step",
        type=float,
        default=None,
        help=(
            "CtrlRegen+ strength in [0, 1]; larger removes more watermark and "
            "changes the image more. Requires --mode ctrlregen-plus (or implies it)."
        ),
    )
    regen_group.add_argument("--seed", type=int, default=0, help="Random seed.")
    regen_group.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Denoising steps for the full reverse trajectory.",
    )
    regen_group.add_argument(
        "--guidance-scale",
        type=float,
        default=2.0,
        help="Text classifier-free-guidance scale.",
    )

    advanced_group = parser.add_argument_group("advanced")
    advanced_group.add_argument(
        "--controlnet-scale",
        type=float,
        default=1.0,
        help="Spatial ControlNet conditioning scale.",
    )
    advanced_group.add_argument(
        "--ip-adapter-scale",
        type=float,
        default=1.0,
        help="Semantic IP-Adapter scale.",
    )
    advanced_group.add_argument(
        "--canny-low",
        type=int,
        default=100,
        help="Lower Canny threshold.",
    )
    advanced_group.add_argument(
        "--canny-high",
        type=int,
        default=150,
        help="Upper Canny threshold.",
    )
    advanced_group.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda, cuda:1, cpu.",
    )
    advanced_group.add_argument(
        "--no-color-match",
        action="store_true",
        help="Disable the post-hoc colour transfer from the input image.",
    )

    args = parser.parse_args(argv)

    # --step is a CtrlRegen+-only option. Supplying it is enough to select that
    # mode; CtrlRegen is the default when no step is given.
    if args.mode is None:
        args.mode = MODE_CTRLREGEN_PLUS if args.step is not None else MODE_CTRLREGEN

    if args.mode == MODE_CTRLREGEN_PLUS:
        if args.step is None:
            parser.error("--step is required with --mode ctrlregen-plus")
        if not 0.0 <= args.step <= 1.0:
            parser.error("--step must be in [0, 1]")
    elif args.step is not None:
        parser.error("--step is only used with --mode ctrlregen-plus")

    if args.num_inference_steps < 1:
        parser.error("--num-inference-steps must be >= 1")
    if args.guidance_scale <= 0:
        parser.error("--guidance-scale must be > 0")
    if args.controlnet_scale < 0:
        parser.error("--controlnet-scale must be >= 0")
    if args.ip_adapter_scale < 0:
        parser.error("--ip-adapter-scale must be >= 0")
    if not 0 <= args.canny_low < args.canny_high <= 255:
        parser.error("Canny thresholds must satisfy 0 <= low < high <= 255")

    try:
        _resolve_device(args.device)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    if not args.input.exists():
        parser.error(f"input not found: {args.input}")

    return args


def main(argv=None):
    args = parse_args(argv)

    try:
        inputs = collect_inputs(args.input)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None

    if not inputs:
        raise SystemExit(f"No images found in {args.input}")
    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise SystemExit(f"output path is not a directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"Images: {len(inputs)}")
    print("Loading models...")
    pipe, canny = load_models(args.device)

    # The IP-Adapter scale is a property of the loaded UNet, not a per-call
    # argument, so apply the CLI value once after loading.
    pipe.set_ip_adapter_scale(args.ip_adapter_scale)

    if args.mode == MODE_CTRLREGEN:
        suffix = f"ctrlregen_seed{args.seed}"
    else:
        suffix = f"ctrlregen-plus_step{args.step:g}_seed{args.seed}"

    common_kwargs = {
        "pipe": pipe,
        "canny": canny,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "controlnet_scale": args.controlnet_scale,
        "canny_low": args.canny_low,
        "canny_high": args.canny_high,
        "match_colors": not args.no_color_match,
    }

    failures = []
    total = len(inputs)
    for index, src in enumerate(inputs, 1):
        print(f"[{index}/{total}] {src.name}")
        try:
            with Image.open(src) as img:
                if args.mode == MODE_CTRLREGEN:
                    output = ctrl_regen(img, seed=args.seed, **common_kwargs)
                else:
                    output = ctrl_regen_plus(
                        img, step=args.step, seed=args.seed, **common_kwargs
                    )
        except Exception as exc:
            failures.append(src)
            print(f"  failed: {exc}", file=sys.stderr)
            continue

        dst = args.output_dir / f"{src.stem}_{suffix}.png"
        output.save(dst)
        print(f"  saved: {dst}")

    if failures:
        print(f"{len(failures)} of {total} image(s) failed:", file=sys.stderr)
        for src in failures:
            print(f"  {src}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
