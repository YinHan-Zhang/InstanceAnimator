export MODEL_NAME="Wan2.1-Fun-14B-Control"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/video_data.json"

NCCL_DEBUG=INFO

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 --gpu_ids 0,1 --main_process_port 29501 training/train_control_lora.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=512 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=2 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --video_repeat=2 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=2000 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="ckpt" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --train_mode="control_ref" \
  --control_ref_image="random" \
  --low_vram  \
  --rank 64 \
  --network_alpha 32 \
  --resume_from_checkpoint "latest"