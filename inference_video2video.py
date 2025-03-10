import os
import numpy as np
import argparse
import imageio
import torch
import cv2

from einops import rearrange
from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from controlnet_aux import CannyDetector

from models.pipeline_controlvideo import ControlVideoToVideoEbsynthPipeline

from models.util import save_videos_grid, read_video, get_annotation
from models.unet import UNet3DConditionModel
from models.controlnet import ControlNetModel3D
from models.RIFE.IFNet_HDv3 import IFNet
from models.video_image_convert import video2imglist, imglist2video, merge_videos


POS_PROMPT = " ,best quality, extremely detailed, HD, ultra-realistic, 8K, HQ, masterpiece, trending on artstation, art, smooth"
NEG_PROMPT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_path", type=str, required=True, help="path of the sd model, in diffusers")
    parser.add_argument("--canny_ctrl_path", type=str, required=True, help="path of the canny control model, in diffusers")
    parser.add_argument("--prompt", type=str, required=True, help="Text description of target video")
    parser.add_argument("--video_path", type=str, required=True, help="Path to a source video")
    parser.add_argument("--output_path", type=str, default="./outputs", help="Directory of output")
    parser.add_argument("--video_length", type=int, default=15, help="Length of synthesized video")
    parser.add_argument("--height", type=int, default=512, help="Height of synthesized video, and should be a multiple of 32")
    parser.add_argument("--width", type=int, default=512, help="Width of synthesized video, and should be a multiple of 32")
    parser.add_argument("--smoother_steps", nargs='+', default=[19, 20], type=int, help="Timesteps at which using interleaved-frame smoother")
    parser.add_argument("--seed", type=int, default=42, help="Random seed of generator")
    parser.add_argument("--num_steps", type=int, default=50, help="num of inference steps")
    parser.add_argument("--strength", type=float, default=0.8, help="strength of noise")
    parser.add_argument("--guidance_scale", type=float, default=12.5, help="guidance_scale of prompt")
    parser.add_argument("--control_scale", type=float, default=1.3, help="control_scale")
    parser.add_argument("--window_size", type=int, default=30, help="window_size")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)
    
    # Height and width should be a multiple of 32
    args.height = (args.height // 32) * 32    
    args.width = (args.width // 32) * 32    

    annotator = CannyDetector()

    tokenizer = CLIPTokenizer.from_pretrained(args.sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_path, subfolder="text_encoder").to(dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(args.sd_path, subfolder="vae").to(dtype=torch.float16)
    unet = UNet3DConditionModel.from_pretrained_2d(args.sd_path, subfolder="unet").to(dtype=torch.float16)
    controlnet = ControlNetModel3D.from_pretrained_2d(args.canny_ctrl_path).to(dtype=torch.float16)
    interpolater = IFNet(ckpt_path="checkpoints/flownet.pkl").to(dtype=torch.float16)
    scheduler=DDIMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")

    pipe = ControlVideoToVideoEbsynthPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet, interpolater=interpolater, scheduler=scheduler,
        )

    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to("cuda")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)

    # Step 1. Read a video 
    video = read_video(video_path=args.video_path, video_length=args.video_length, width=args.width, height=args.height)

    video_images = []
    for fi in range(video.size()[0]):
        video_images.append(((video[fi].permute(1,2,0)+1)*127.5).cpu().numpy())
    
    # Save source video
    original_pixels = rearrange(video, "(b f) c h w -> b c f h w", b=1)
    save_videos_grid(original_pixels, os.path.join(args.output_path, "source_video.mp4"), rescale=True)


    # Step 2. Parse a video to canny frames
    pil_annotation = get_annotation(video, annotator)

    # Save canny video
    video_cond = [np.array(p).astype(np.uint8) for p in pil_annotation]
    imageio.mimsave(os.path.join(args.output_path, f"canny.mp4"), video_cond, fps=8)

    # Reduce memory (optional)
    del annotator; torch.cuda.empty_cache()

    # Step 3. inference
    sample = pipe(args.prompt+POS_PROMPT, video_length=args.video_length, video_images=video_images, frames=pil_annotation, 
                strength=args.strength, num_inference_steps=args.num_steps, smooth_steps=args.smoother_steps,
                generator=generator, guidance_scale=args.guidance_scale, negative_prompt=NEG_PROMPT,
                width=args.width, height=args.height, controlnet_conditioning_scale=args.control_scale
            ).videos
    save_videos_grid(sample, f"{args.output_path}/{os.path.basename(args.video_path)}")
    merge_videos(os.path.join(args.output_path, "source_video.mp4"), f"{args.output_path}/{os.path.basename(args.video_path)}", f"{args.output_path}/combine_{os.path.basename(args.video_path)}")