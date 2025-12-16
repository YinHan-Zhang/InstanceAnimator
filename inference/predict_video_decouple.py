import os
import sys

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from src.dist import set_multi_gpus_devices, shard_model
from src.models import (AutoencoderKLWan, AutoTokenizer, CLIPModel,
                               WanT5EncoderModel, WanTransformer3DModel)
from src.data.dataset_image_video import process_pose_file
from src.models.cache_utils import get_teacache_coefficients
from src.pipeline import WanPipeline
from src.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from src.utils.lora_utils import merge_lora, unmerge_lora
from src.utils.utils import (filter_kwargs, get_image_to_video_latent, get_image_latent, get_video_to_video_latent, save_videos_grid)
from src.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from src.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

GPU_memory_mode     = None

ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
fsdp_text_encoder   = False
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# Support TeaCache.
enable_teacache     = False
teacache_threshold  = 0.10
# The number of steps to skip TeaCache at the beginning of the inference process, which can
# reduce the impact of TeaCache on generated video quality.
num_skip_start_steps = 5
# Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
teacache_offload    = False

# Skip some cfg steps in inference for acceleration
# Recommended to be set between 0.00 and 0.25
cfg_skip_ratio      = 0

# Riflex config
enable_riflex       = False
# Index of intrinsic frequency
riflex_k            = 6

# Config and model path
config_path         = "config/wan2.1/wan_civitai.yaml"
# model path
model_name          = "Wan2.1-Fun-14B-Control"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics. 
# Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
# If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
# If you want to generate a 720p video, it is recommended to set the shift value to 5.0.
shift               = 3

# Load pretrained model if need
transformer_path    = "transformer.safetensors"
vae_path            = None
lora_path           = "lora.safetensors"

# Other params
sample_size         = [320, 512]
video_length        = 49
fps                 = 16

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16
control_video           = "sketch.mp4"
bg_image                = "bg.jpeg"

subject_ref_images      = [
    "instance1.png",
    "instance2.png",
]

prompt                  = "your prompt here."

negative_prompt         = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


guidance_scale          = 3.0
seed                    = 43
num_inference_steps     = 30
lora_weight             = 0.85
save_path               = "samples/video"
text_control_strength   = 0.5
bg_control_strength     = 0.5
ins_control_strength    = 0.4

device = "cuda:0"
config = OmegaConf.load(config_path)

transformer = WanTransformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    transformer_path=transformer_path,
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# Get Text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Get Clip Image Encoder
clip_image_encoder = CLIPModel.from_pretrained(
    os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
).to(weight_dtype)
clip_image_encoder = clip_image_encoder.eval()

# Get Scheduler
Chosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
    config['scheduler_kwargs']['shift'] = 1
scheduler = Chosen_Scheduler(
    **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline
pipeline = WanPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
    clip_image_encoder=clip_image_encoder
)
if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )

if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)

def save_results(name):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    if video_length == 1:
        video_path = os.path.join(save_path, name + ".png")

        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(video_path)
    else:
        video_path = os.path.join(save_path, name + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)


with torch.no_grad():
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1
    
    if enable_riflex:
        pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)
    
    if bg_image is not None:
        clip_image = [ Image.open(bg_image).convert("RGB") ]
        for _subject_ref_image in subject_ref_images:
            clip_image.append(Image.open(_subject_ref_image).convert("RGB"))
    else:
        clip_image = [ Image.open(bg_image).convert("RGB") ]

    if bg_image is not None:
        bg_image = Image.open(bg_image).convert("RGB")
        bg_image = get_image_latent(bg_image, sample_size=sample_size)
    else:
        bg_image = None

    if subject_ref_images is not None:
        subject_ref_images = [get_image_latent(_subject_ref_image, sample_size=sample_size, padding=True) for _subject_ref_image in subject_ref_images]

    input_video, input_video_mask, _, _ = get_video_to_video_latent(control_video, None, video_length=video_length, sample_size=sample_size, fps=fps, ref_image=None)

    name = "result"
    sample = pipeline(
        prompt, 
        num_frames = video_length,
        negative_prompt = negative_prompt,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
        guidance_scale = guidance_scale,
        num_inference_steps = num_inference_steps,

        control_video = input_video,
        subject_ref_images = subject_ref_images,
        clip_image = clip_image,
        shift = shift,
        bg_image=bg_image,
        text_control_strength=text_control_strength,
        bg_control_strength=bg_control_strength,
        ins_control_strength=ins_control_strength,
    ).videos

    save_results(name)

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device, dtype=weight_dtype)
