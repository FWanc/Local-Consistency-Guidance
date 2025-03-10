# eval "$(conda shell.bash hook)"
# conda activate controlvideo

python inference_video2video.py \
    --sd_path "./dreambooth_models/SD1.5_25elsa" \
    --canny_ctrl_path "path/to/your/models--lllyasviel--sd-controlnet-canny" \
    --prompt "photo of a sks gril" \
    --video_path "path/to/your/input/video" \
    --output_path "path/to/your/output/video" \
    --num_steps 100 \
    --strength 0.5 \
    --control_scale 1.5 \
    --guidance_scale 10.5 \
    --video_length 30 \
    --smoother_steps 14 15 \
    --width 512 \
    --height 512 \