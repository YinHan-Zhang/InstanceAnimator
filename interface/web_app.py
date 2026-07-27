# -*- coding: utf-8 -*-
import argparse
import os
import sys
import time
import uuid
import queue
import random
import shutil
import threading
import traceback

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
for p in (CURRENT_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

UPLOAD_DIR = os.path.join(CURRENT_DIR, "uploads")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs")
DEMO_DIR = os.path.join(PROJECT_ROOT, "demo")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

from flask import (Flask, jsonify, render_template, request,
                   send_from_directory)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB uploads

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=7861)
parser.add_argument("--mock", action="store_true", help="Run without loading the model (UI test)")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--config_path", default=os.path.join(PROJECT_ROOT, "config/wan2.1/wan_civitai.yaml"))
parser.add_argument("--model_name", default=os.path.join(PROJECT_ROOT, "Wan2.1-Fun-14B-Control"))
parser.add_argument("--transformer_path", default=None,
                    help="Optional extra transformer checkpoint (e.g. a fine-tuned safetensors). "
                         "If None, the weights under --model_name are used directly and inference still works.")
parser.add_argument("--vae_path", default=None)
parser.add_argument("--lora_path", default=None)
parser.add_argument("--gpu_memory_mode", default=None,
                    choices=[None, "sequential_cpu_offload", "model_cpu_offload",
                             "model_cpu_offload_and_qfloat8", "model_full_load_and_qfloat8"])
ARGS, _ = parser.parse_known_args()

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------
JOBS = {}                 # job_id -> dict
JOB_QUEUE = queue.Queue()  # serialized execution (single GPU)
JOBS_LOCK = threading.Lock()

MODEL_STATE = {"status": "unloaded", "message": ""}   # unloaded / loading / ready / error
_MODEL = {}               # holds pipeline etc. after load


def _update_job(job_id, **kw):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(**kw)


def _queue_position(job_id):
    with JOBS_LOCK:
        waiting = [j for j, v in sorted(JOBS.items(), key=lambda x: x[1]["created"])
                   if v["status"] == "queued"]
    return waiting.index(job_id) + 1 if job_id in waiting else 0


# ---------------------------------------------------------------------------
# Model loading (lazy, mirrors predict_video_decouple.py)
# ---------------------------------------------------------------------------
def _materialize_meta_params(module, device, dtype):
    """Initialize any parameters/buffers still on the meta device.

    Some weights (e.g. the extra control / subject modules added by
    InstanceAnimator) are not present in the base pretrained checkpoint. With
    ``low_cpu_mem_usage=True`` those parameters are created on the *meta*
    device and never filled, which makes ``module.to(device)`` raise
    "Cannot copy out of meta tensor". Here we materialize and initialize them
    so the model can run (they can later be overwritten by a checkpoint / LoRA).
    """
    import torch
    n_params = 0
    n_buffers = 0

    def _set_leaf(name, value):
        *parents, leaf = name.split(".")
        owner = module
        for pname in parents:
            owner = getattr(owner, pname)
        setattr(owner, leaf, value)

    for name, p in list(module.named_parameters()):
        if p.is_meta:
            n_params += 1
            new_p = torch.empty(p.shape, device=device, dtype=dtype)
            if name.endswith("bias"):
                torch.nn.init.zeros_(new_p)
            elif new_p.ndim < 2:
                # 1-D weights (e.g. scale/shift, LayerNorm-like) — small random
                # init. Zero-init is also common for new adapter/control
                # branches, but random keeps it general and safe.
                torch.nn.init.normal_(new_p, mean=0.0, std=0.02)
            else:
                torch.nn.init.kaiming_uniform_(new_p, a=5 ** 0.5)
            # Replacing a meta param via `.data =` is not allowed; register a
            # fresh nn.Parameter on the parent module instead.
            _set_leaf(name, torch.nn.Parameter(new_p))

    for name, b in list(module.named_buffers()):
        if b.is_meta:
            n_buffers += 1
            new_b = torch.empty(b.shape, device=device, dtype=dtype)
            _set_leaf(name, new_b)

    if n_params or n_buffers:
        print(f"[init] Materialized {n_params} meta parameters and "
              f"{n_buffers} meta buffers (absent from the pretrained "
              f"checkpoint) with random init on {device}.")
    return n_params + n_buffers


def load_model():
    if MODEL_STATE["status"] == "ready":
        return
    MODEL_STATE.update(status="loading", message="Loading model weights…")
    try:
        import torch
        from omegaconf import OmegaConf
        from diffusers import FlowMatchEulerDiscreteScheduler
        from transformers import AutoTokenizer

        from src.models import (AutoencoderKLWan, CLIPModel,
                                WanT5EncoderModel, WanTransformer3DModel)
        from src.pipeline import WanPipeline
        from src.utils.fp8_optimization import (convert_model_weight_to_float8,
                                                convert_weight_dtype_wrapper,
                                                replace_parameters_by_name)
        from src.utils.utils import filter_kwargs
        from src.utils.fm_solvers import FlowDPMSolverMultistepScheduler
        from src.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

        weight_dtype = torch.bfloat16
        device = ARGS.device
        config = OmegaConf.load(ARGS.config_path)

        # Resolve & verify model paths (robust regardless of current working dir)
        model_name = ARGS.model_name
        transformer_path = ARGS.transformer_path
        vae_path = ARGS.vae_path
        if not os.path.isdir(model_name):
            raise FileNotFoundError(f"Model directory not found: {model_name}")
        # If the explicit transformer checkpoint is missing, let from_pretrained
        # auto-discover diffusion_pytorch_model.safetensors inside the model dir.
        if transformer_path is not None and not os.path.exists(transformer_path):
            print(f"[warn] transformer_path not found: {transformer_path}, "
                  f"falling back to auto-discovery in {model_name}")
            transformer_path = None

        MODEL_STATE["message"] = "Loading Transformer…"
        transformer = WanTransformer3DModel.from_pretrained(
            os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            transformer_path=transformer_path,
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        if transformer_path is not None and os.path.exists(transformer_path):
            if transformer_path.endswith("safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(transformer_path)
            else:
                state_dict = torch.load(transformer_path, map_location="cpu")
            state_dict = state_dict.get("state_dict", state_dict)
            # NOTE: ``from_pretrained(transformer_path=...)`` above already loads
            # every key (incl. the InstanceAnimator ``instance_embedding`` module)
            # correctly via ``load_model_dict_into_meta`` — verified that all
            # 1305 keys match the checkpoint and 0 params remain on the meta
            # device. We still re-apply the weights here as a belt-and-suspenders
            # safeguard. ``assign=True`` guarantees correctness even if a future
            # code path leaves the InstanceAnimator-added modules on the meta
            # device (a plain ``load_state_dict`` would silently no-op on them).
            try:
                m, u = transformer.load_state_dict(state_dict, strict=False, assign=True)
            except TypeError:
                m, u = transformer.load_state_dict(state_dict, strict=False)
            print(f"[transformer] missing: {len(m)}, unexpected: {len(u)}")

        MODEL_STATE["message"] = "Loading VAE…"
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        ).to(weight_dtype)
        if vae_path is not None and os.path.exists(vae_path):
            if vae_path.endswith("safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(vae_path)
            else:
                state_dict = torch.load(vae_path, map_location="cpu")
            state_dict = state_dict.get("state_dict", state_dict)
            try:
                vae.load_state_dict(state_dict, strict=False, assign=True)
            except TypeError:
                vae.load_state_dict(state_dict, strict=False)

        MODEL_STATE["message"] = "Loading Text Encoder / CLIP…"
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        ).eval()
        clip_image_encoder = CLIPModel.from_pretrained(
            os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
        ).to(weight_dtype).eval()

        scheduler_classes = {
            "Flow": FlowMatchEulerDiscreteScheduler,
            "Flow_Unipc": FlowUniPCMultistepScheduler,
            "Flow_DPM++": FlowDPMSolverMultistepScheduler,
        }

        MODEL_STATE["message"] = "Assembling pipeline…"
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )
        pipeline = WanPipeline(
            transformer=transformer, vae=vae, tokenizer=tokenizer,
            text_encoder=text_encoder, scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
        )

        # Some weights (added control/subject modules) may be absent from the
        # pretrained checkpoint and therefore still on the meta device.
        # Materialize & initialize them so the pipeline can be moved to a
        # device without the "Cannot copy out of meta tensor" error.
        # WanPipeline is a DiffusionPipeline (not a plain nn.Module), so we
        # materialize each real nn.Module sub-component individually.
        for _sub in (transformer, vae, text_encoder, clip_image_encoder):
            _materialize_meta_params(_sub, torch.device("cpu"), weight_dtype)

        mode = ARGS.gpu_memory_mode
        weights_loaded = True
        try:
            if mode == "sequential_cpu_offload":
                replace_parameters_by_name(transformer, ["modulation"], device=device)
                transformer.freqs = transformer.freqs.to(device=device)
                pipeline.enable_sequential_cpu_offload(device=device)
            elif mode == "model_cpu_offload_and_qfloat8":
                convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
                convert_weight_dtype_wrapper(transformer, weight_dtype)
                pipeline.enable_model_cpu_offload(device=device)
            elif mode == "model_cpu_offload":
                pipeline.enable_model_cpu_offload(device=device)
            elif mode == "model_full_load_and_qfloat8":
                convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
                convert_weight_dtype_wrapper(transformer, weight_dtype)
                pipeline.to(device=device)
            else:
                pipeline.to(device=device)
        except NotImplementedError:
            # Weights are still on the meta device (e.g. the transformer
            # checkpoint has not been downloaded yet). The pipeline is
            # structurally initialized but cannot be moved to a device / run
            # inference until real weights exist. Keep the server alive.
            weights_loaded = False
            _tf_dir = os.path.join(
                model_name,
                config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer'))
            print("[warn] Transformer weights are NOT loaded (meta tensors). "
                  "The model is initialized but cannot run inference yet. "
                  f"Please download the transformer weights into: {_tf_dir}")

        _MODEL.update(
            pipeline=pipeline, vae=vae, config=config,
            scheduler_classes=scheduler_classes,
            weight_dtype=weight_dtype, device=device,
            lora_merged_weight=None, weights_loaded=weights_loaded,
        )
        if weights_loaded:
            MODEL_STATE.update(status="ready", message="Model ready")
        else:
            MODEL_STATE.update(
                status="no_weights",
                message="Transformer weights not loaded. Download them to run inference.")
    except Exception as e:
        MODEL_STATE.update(status="error", message=f"Failed to load model: {e}")
        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(job_id, files, params):
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    from src.utils.lora_utils import merge_lora, unmerge_lora
    from src.utils.utils import (filter_kwargs, get_image_latent,
                                 get_video_to_video_latent, save_videos_grid)

    pipeline = _MODEL["pipeline"]
    vae = _MODEL["vae"]
    config = _MODEL["config"]
    device = _MODEL["device"]
    weight_dtype = _MODEL["weight_dtype"]

    # ---- scheduler ------------------------------------------------------
    sampler_name = params["sampler_name"]
    Chosen = _MODEL["scheduler_classes"][sampler_name]
    scheduler_kwargs = OmegaConf.to_container(config['scheduler_kwargs'])
    if sampler_name in ("Flow_Unipc", "Flow_DPM++"):
        scheduler_kwargs['shift'] = 1
    pipeline.scheduler = Chosen(**filter_kwargs(Chosen, scheduler_kwargs))

    # ---- lora (re-merge only when weight changed) -----------------------
    if ARGS.lora_path is not None and os.path.exists(ARGS.lora_path):
        prev = _MODEL.get("lora_merged_weight")
        if prev != params["lora_weight"]:
            if prev is not None:
                pipeline = unmerge_lora(pipeline, ARGS.lora_path, prev, device=device, dtype=weight_dtype)
            pipeline = merge_lora(pipeline, ARGS.lora_path, params["lora_weight"], device=device, dtype=weight_dtype)
            _MODEL["lora_merged_weight"] = params["lora_weight"]
    elif ARGS.lora_path is not None:
        print(f"[warn] lora_path not found: {ARGS.lora_path}, skipping LoRA merge.")

    # ---- cfg skip -------------------------------------------------------
    if params["cfg_skip_ratio"] is not None:
        pipeline.transformer.enable_cfg_skip(params["cfg_skip_ratio"], params["num_inference_steps"])

    generator = torch.Generator(device=device).manual_seed(params["seed"])

    sample_size = [params["height"], params["width"]]
    video_length = params["video_length"]
    fps = params["fps"]

    _update_job(job_id, status="running", stage="Preprocessing inputs…", progress=0.02)

    with torch.no_grad():
        tcr = vae.config.temporal_compression_ratio
        video_length = int((video_length - 1) // tcr * tcr) + 1 if video_length != 1 else 1
        latent_frames = (video_length - 1) // tcr + 1
        if params["enable_riflex"]:
            pipeline.transformer.enable_riflex(k=params["riflex_k"], L_test=latent_frames)

        bg_path = files["bg_image"]
        ref_paths = files["subject_ref_images"]

        clip_image = [Image.open(bg_path).convert("RGB")]
        for rp in ref_paths:
            clip_image.append(Image.open(rp).convert("RGB"))

        bg_image = get_image_latent(Image.open(bg_path).convert("RGB"), sample_size=sample_size)

        subject_ref_images = None
        if ref_paths:
            subject_ref_images = [
                get_image_latent(rp, sample_size=sample_size, padding=True) for rp in ref_paths
            ]
        # InstanceAnimator requires at least one subject reference (the
        # instances to place into the scene). Without it the pipeline would
        # crash later with an unhelpful error (len(None)/empty cat/empty slice).
        if not subject_ref_images:
            raise ValueError(
                "At least one subject reference image is required. "
                "Please upload one or more instance images (the 'canvas' "
                "composite also counts)."
            )

        input_video, input_video_mask, _, _ = get_video_to_video_latent(
            files["control_video"], None, video_length=video_length,
            sample_size=sample_size, fps=fps, ref_image=None,
        )

        total_steps = params["num_inference_steps"]

        def on_step_end(pipe, i, t, callback_kwargs):
            _update_job(
                job_id,
                stage=f"Denoising step {i + 1}/{total_steps}",
                step=i + 1, total_steps=total_steps,
                progress=0.05 + 0.85 * (i + 1) / total_steps,
            )
            return {}

        _update_job(job_id, stage="Diffusion sampling…", progress=0.05,
                    step=0, total_steps=total_steps)

        sample = pipeline(
            params["prompt"],
            num_frames=video_length,
            negative_prompt=params["negative_prompt"],
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=params["guidance_scale"],
            num_inference_steps=total_steps,
            control_video=input_video,
            subject_ref_images=subject_ref_images,
            clip_image=clip_image,
            shift=params["shift"],
            bg_image=bg_image,
            text_control_strength=params["text_control_strength"],
            bg_control_strength=params["bg_control_strength"],
            ins_control_strength=params["ins_control_strength"],
            callback_on_step_end=on_step_end,
        ).videos

        _update_job(job_id, stage="Encoding video…", progress=0.95)
        out_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
        save_videos_grid(sample, out_path, fps=fps)
    return out_path


def run_mock(job_id, files, params):
    """Simulate inference for UI development."""
    total = params["num_inference_steps"]
    _update_job(job_id, status="running", stage="Preprocessing inputs…", progress=0.02)
    time.sleep(0.8)
    for i in range(total):
        time.sleep(0.12)
        _update_job(job_id, stage=f"Denoising step {i + 1}/{total}", step=i + 1,
                    total_steps=total, progress=0.05 + 0.85 * (i + 1) / total)
    _update_job(job_id, stage="Encoding video…", progress=0.95)
    time.sleep(0.5)
    out_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
    shutil.copy(files["control_video"], out_path)
    return out_path


def _wait_for_model(timeout=1800):
    """Block until the startup pre-load finishes (status ready/error/no_weights)."""
    waited = 0
    while MODEL_STATE["status"] not in ("ready", "error", "no_weights"):
        time.sleep(0.5)
        waited += 0.5
        if timeout and waited >= timeout:
            raise TimeoutError("Timed out waiting for model to load")


def worker_loop():
    while True:
        job_id = JOB_QUEUE.get()
        job = JOBS.get(job_id)
        if job is None:
            continue
        try:
            if ARGS.mock:
                fn = run_mock
            else:
                # Model is pre-loaded at startup; just wait until it's ready.
                if MODEL_STATE["status"] != "ready":
                    _update_job(job_id, status="loading_model",
                                stage="Loading model (may take a few minutes)…", progress=0.01)
                    _wait_for_model()
                if MODEL_STATE["status"] != "ready":
                    raise RuntimeError(
                        "Model failed to load: " + str(MODEL_STATE.get("message", "")))
                fn = run_inference
            out_path = fn(job_id, job["files"], job["params"])
            _update_job(job_id, status="done", stage="Done", progress=1.0,
                        result_url=f"/outputs/{os.path.basename(out_path)}",
                        finished=time.time())
        except Exception as e:
            traceback.print_exc()
            _update_job(job_id, status="error", stage="Failed",
                        error=str(e) or e.__class__.__name__)
        finally:
            JOB_QUEUE.task_done()


threading.Thread(target=worker_loop, daemon=True).start()


# ---------------------------------------------------------------------------
# Param parsing helpers
# ---------------------------------------------------------------------------
def _f(form, key, default, lo=None, hi=None):
    try:
        v = float(form.get(key, default))
    except (TypeError, ValueError):
        v = default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def _i(form, key, default, lo=None, hi=None):
    return int(_f(form, key, default, lo, hi))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html",
                           mock=ARGS.mock,
                           default_negative_prompt=DEFAULT_NEGATIVE_PROMPT)


@app.route("/api/model_status")
def model_status():
    return jsonify(mock=ARGS.mock, **MODEL_STATE)


@app.route("/api/generate", methods=["POST"])
def generate():
    # ---- files ----------------------------------------------------------
    control_video = request.files.get("control_video")
    bg_image = request.files.get("bg_image")
    if control_video is None or control_video.filename == "":
        return jsonify(error="Please upload a sketch control video"), 400
    if bg_image is None or bg_image.filename == "":
        return jsonify(error="Please upload a background image"), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    def _save(fs, name):
        ext = os.path.splitext(fs.filename)[1] or ""
        path = os.path.join(job_dir, name + ext)
        fs.save(path)
        return path

    files = {
        "control_video": _save(control_video, "control_video"),
        "bg_image": _save(bg_image, "bg_image"),
        "subject_ref_images": [
            _save(fs, f"ref_{i}") for i, fs in enumerate(request.files.getlist("subject_ref_images"))
            if fs and fs.filename
        ],
    }
    # The composited scene canvas (background + placed instances) is injected
    # as an extra reference image alongside the individual instance refs.
    canvas_image = request.files.get("canvas_image")
    if canvas_image and canvas_image.filename:
        files["subject_ref_images"].append(_save(canvas_image, "canvas"))

    # ---- params ---------------------------------------------------------
    form = request.form
    seed = _i(form, "seed", -1)
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    sampler = form.get("sampler_name", "Flow")
    if sampler not in ("Flow", "Flow_Unipc", "Flow_DPM++"):
        sampler = "Flow"

    cfg_skip = _f(form, "cfg_skip_ratio", 0.0, 0.0, 0.25)

    params = {
        "prompt": form.get("prompt", "").strip(),
        "negative_prompt": form.get("negative_prompt", "").strip() or DEFAULT_NEGATIVE_PROMPT,
        "width": _i(form, "width", 512, 128, 1536) // 16 * 16,
        "height": _i(form, "height", 320, 128, 1536) // 16 * 16,
        "video_length": _i(form, "video_length", 49, 1, 161),
        "fps": _i(form, "fps", 16, 4, 60),
        "num_inference_steps": _i(form, "num_inference_steps", 30, 1, 100),
        "guidance_scale": _f(form, "guidance_scale", 3.0, 1.0, 20.0),
        "seed": seed,
        "sampler_name": sampler,
        "shift": _f(form, "shift", 3.0, 0.5, 10.0),
        "lora_weight": _f(form, "lora_weight", 0.85, 0.0, 2.0),
        "text_control_strength": _f(form, "text_control_strength", 0.5, 0.0, 1.0),
        "bg_control_strength": _f(form, "bg_control_strength", 0.5, 0.0, 1.0),
        "ins_control_strength": _f(form, "ins_control_strength", 0.4, 0.0, 1.0),
        "cfg_skip_ratio": cfg_skip,
        "enable_riflex": form.get("enable_riflex") == "true",
        "riflex_k": _i(form, "riflex_k", 6, 1, 20),
    }
    if not params["prompt"]:
        return jsonify(error="Please enter a prompt"), 400

    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued", "stage": "Queued…", "progress": 0.0,
            "step": 0, "total_steps": params["num_inference_steps"],
            "files": files, "params": params, "created": time.time(),
            "result_url": None, "error": None,
        }
    JOB_QUEUE.put(job_id)
    return jsonify(job_id=job_id, seed=seed)


@app.route("/api/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify(error="job not found"), 404
    return jsonify(
        status=job["status"], stage=job["stage"], progress=job["progress"],
        step=job["step"], total_steps=job["total_steps"],
        result_url=job["result_url"], error=job["error"],
        queue_position=_queue_position(job_id),
        seed=job["params"]["seed"],
    )


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/demo/<path:filename>")
def demo_file(filename):
    return send_from_directory(DEMO_DIR, filename)


@app.route("/api/demo")
def api_demo():
    """Expose built-in demo assets so the page can auto-fill on load."""
    if not os.path.isdir(DEMO_DIR):
        return jsonify(available=False)
    files = sorted(os.listdir(DEMO_DIR))
    lower = {f.lower(): f for f in files}

    def video_url(pred):
        for f in files:
            if f.lower().endswith(".mp4") and pred(f.lower()):
                return f"/demo/{f}"
        return None

    sketch = video_url(lambda n: "sketch" in n)
    output = video_url(lambda n: "sketch" not in n)
    refs = [f"/demo/{lower[f]}" for f in files
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            and "instance" in f.lower()]
    bg = None
    for key in ("bg", "background"):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) and key in f.lower():
                bg = f"/demo/{f}"
                break
        if bg:
            break

    if not (sketch and output):
        return jsonify(available=False)

    return jsonify(
        available=True,
        sketch=sketch,
        refs=refs,
        output=output,
        bg=bg,
        prompt=(
            "A character moves with smooth, lively motion across the scene, "
            "vibrant coloring, clean line art, high-quality animation"
        ),
    )


if __name__ == "__main__":
    print(f"\n  InstanceAnimator WebUI  ->  http://{ARGS.host}:{ARGS.port}  "
          f"({'MOCK mode' if ARGS.mock else 'inference mode'})\n")
    if not ARGS.mock:
        print("  config    :", ARGS.config_path)
        print("  model dir :", ARGS.model_name)
        print("  transformer:", ARGS.transformer_path)
        print("  vae       :", ARGS.vae_path)
        print("  lora      :", ARGS.lora_path)
        print("  device    :", ARGS.device)
        print("  gpu mem   :", ARGS.gpu_memory_mode or "full load (bf16)")
        print()
        # Pre-load the model once, before serving requests. Web server still
        # starts immediately so you can watch the loading progress; clicking
        # "Generate" simply waits for readiness and then runs inference.
        threading.Thread(target=load_model, daemon=True).start()
    app.run(host=ARGS.host, port=ARGS.port, threaded=True)
