# InstanceAnimator: Multi-Instance Sketch Video Colorization


![](./assets/logo.jpeg)


# News

- 2025-012-16 : release code ✅

# Set up

## Environment

    conda create -n InstanceAnimator python=3.12

    pip install -r requirements.txt

## Repository

    git clone https://github.com/YinHan-Zhang/InstanceAnimator.git
    
    cd InstanceAnimator


## OpenAnimate Dataset

    modelscope login
    modelscope download --dataset NiceYinHan/OpenAnimate --local_dir ./OpenAnimate


# Train

Dataset Format:

```json
{
   "file_path": "video.mp4",
    "sketch_file_path": "sketch.mp4",
    "control_file_path": [
        "instance_1.jpg",
        ...
    ],
    "background_path": "background.jpg",
    "text": "",
    "type": "video"
}

```bash
    bash training/train_control_lora.sh
```

# inference

```bash
    python  inference/predict_video_decouple.py
```


# Limitation

Due to the limitations of computing resources and data, the amount of data for model training is limited. If you have enough computing resources, you can train the model yourself.

# Acknowledgement

Thanks for the reference contributions of these works: 
    - VideoX-Fun