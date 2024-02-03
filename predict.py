import os
from cog import BasePredictor, Input, Path
import sys
sys.path.append('/content/AnimateLCM-hf')

# commands = [
#     'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/wangfuyun/AnimateLCM/resolve/main/models/DreamBooth_LoRA/cartoon3d.safetensors -d /content/AnimateLCM-hf/models/DreamBooth_LoRA -o cartoon3d.safetensors',
#     'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/wangfuyun/AnimateLCM/resolve/main/models/DreamBooth_LoRA/realistic1.safetensors -d /content/AnimateLCM-hf/models/DreamBooth_LoRA -o realistic1.safetensors',
#     'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/wangfuyun/AnimateLCM/resolve/main/models/DreamBooth_LoRA/realistic2.safetensors -d /content/AnimateLCM-hf/models/DreamBooth_LoRA -o realistic2.safetensors',
#     'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/wangfuyun/AnimateLCM/resolve/main/models/LCM_LoRA/sd15_t2v_beta_lora.safetensors -d /content/AnimateLCM-hf/models/LCM_LoRA -o sd15_t2v_beta_lora.safetensors',
#     'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/wangfuyun/AnimateLCM/resolve/main/models/Motion_Module/sd15_t2v_beta_motion.ckpt -d /content/AnimateLCM-hf/models/Motion_Module -o sd15_t2v_beta_motion.ckpt',
#     'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/wangfuyun/AnimateLCM/resolve/main/models/StableDiffusion/stable-diffusion-v1-5/text_encoder/model.safetensors -d /content/AnimateLCM-hf/models/StableDiffusion/stable-diffusion-v1-5/text_encoder -o model.safetensors',
#     'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/wangfuyun/AnimateLCM/resolve/main/models/StableDiffusion/stable-diffusion-v1-5/unet/diffusion_pytorch_model.bin -d /content/AnimateLCM-hf/models/StableDiffusion/stable-diffusion-v1-5/unet -o diffusion_pytorch_model.bin',
#     'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/wangfuyun/AnimateLCM/resolve/main/models/StableDiffusion/stable-diffusion-v1-5/vae/diffusion_pytorch_model.bin -d /content/AnimateLCM-hf/models/StableDiffusion/stable-diffusion-v1-5/vae -o diffusion_pytorch_model.bin'
# ]

# for command in commands:
#     os.system(command)

import json
import torch
import random

from glob import glob
from omegaconf import OmegaConf
from datetime import datetime
from safetensors import safe_open

from diffusers import AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from animatelcm.scheduler.lcm_scheduler import LCMScheduler
from animatelcm.models.unet import UNet3DConditionModel
from animatelcm.pipelines.pipeline_animation import AnimationPipeline
from animatelcm.utils.util import save_videos_grid
from animatelcm.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatelcm.utils.convert_lora_safetensor_to_diffusers import convert_lora
from animatelcm.utils.lcm_utils import convert_lcm_lora
import copy

sample_idx = 0
scheduler_dict = {
    "LCM": LCMScheduler,
}

class AnimateController:
    def __init__(self):

        # config dirs
        self.basedir = os.getcwd()
        self.stable_diffusion_dir = os.path.join(
            self.basedir, "models", "StableDiffusion")
        self.motion_module_dir = os.path.join(
            self.basedir, "models", "Motion_Module")
        self.personalized_model_dir = os.path.join(
            self.basedir, "models", "DreamBooth_LoRA")
        self.savedir = os.path.join(
            self.basedir, "samples", datetime.now().strftime("Gradio-%Y-%m-%dT%H-%M-%S"))
        self.savedir_sample = os.path.join(self.savedir, "sample")
        self.lcm_lora_path = "models/LCM_LoRA/sd15_t2v_beta_lora.safetensors"
        os.makedirs(self.savedir, exist_ok=True)

        self.stable_diffusion_list = []
        self.motion_module_list = []
        self.personalized_model_list = []

        self.refresh_stable_diffusion()
        self.refresh_motion_module()
        self.refresh_personalized_model()

        # config models
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.pipeline = None
        self.lora_model_state_dict = {}

        self.inference_config = OmegaConf.load("configs/inference.yaml")

    def refresh_stable_diffusion(self):
        self.stable_diffusion_list = glob(
            os.path.join(self.stable_diffusion_dir, "*/"))

    def refresh_motion_module(self):
        motion_module_list = glob(os.path.join(
            self.motion_module_dir, "*.ckpt"))
        self.motion_module_list = [
            os.path.basename(p) for p in motion_module_list]

    def refresh_personalized_model(self):
        personalized_model_list = glob(os.path.join(
            self.personalized_model_dir, "*.safetensors"))
        self.personalized_model_list = [
            os.path.basename(p) for p in personalized_model_list]

    def update_stable_diffusion(self, stable_diffusion_dropdown):
        stable_diffusion_dropdown = os.path.join(self.stable_diffusion_dir,stable_diffusion_dropdown)
        self.tokenizer = CLIPTokenizer.from_pretrained(stable_diffusion_dropdown, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(stable_diffusion_dropdown, subfolder="text_encoder").cuda()
        self.vae = AutoencoderKL.from_pretrained(stable_diffusion_dropdown, subfolder="vae").cuda()
        self.unet = UNet3DConditionModel.from_pretrained_2d(stable_diffusion_dropdown, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(self.inference_config.unet_additional_kwargs)).cuda()

    def update_motion_module(self, motion_module_dropdown):
        motion_module_dropdown = os.path.join(self.motion_module_dir, motion_module_dropdown)
        motion_module_state_dict = torch.load(motion_module_dropdown, map_location="cpu")
        missing, unexpected = self.unet.load_state_dict(motion_module_state_dict, strict=False)

    def update_base_model(self, base_model_dropdown):
        base_model_dropdown = os.path.join(self.personalized_model_dir, base_model_dropdown)
        base_model_state_dict = {}
        with safe_open(base_model_dropdown, framework="pt", device="cpu") as f:
            for key in f.keys():
                base_model_state_dict[key] = f.get_tensor(key)
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(
            base_model_state_dict, self.vae.config)
        self.vae.load_state_dict(converted_vae_checkpoint)
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_model_state_dict, self.unet.config)
        self.unet.load_state_dict(converted_unet_checkpoint, strict=False)

    def update_lora_model(self, lora_model_dropdown):
        lora_model_dropdown = os.path.join(
            self.personalized_model_dir, lora_model_dropdown)
        self.lora_model_state_dict = {}
        if lora_model_dropdown == "none":
            pass
        else:
            with safe_open(lora_model_dropdown, framework="pt", device="cpu") as f:
                for key in f.keys():
                    self.lora_model_state_dict[key] = f.get_tensor(key)

def inference(pipeline, prompt, negative_prompt, num_inference_steps, guidance_scale, width, height, video_length):
    torch.seed()
    seed = torch.initial_seed()

    sample = pipeline(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                video_length=video_length,
            ).videos

    save_sample_path = os.path.join(controller.savedir_sample, f"{sample_idx}.mp4")
    save_videos_grid(sample, save_sample_path)
    return save_sample_path

class Predictor(BasePredictor):
    def setup(self) -> None:
        controller = AnimateController()
        controller.update_stable_diffusion("/content/AnimateLCM-hf/models/StableDiffusion/stable-diffusion-v1-5")
        controller.update_motion_module("/content/AnimateLCM-hf/models/Motion_Module/sd15_t2v_beta_motion.ckpt")
        controller.update_base_model("/content/AnimateLCM-hf/models/DreamBooth_LoRA/cartoon3d.safetensors")
        # controller.update_lora_model("")

        if is_xformers_available():
            controller.unet.enable_xformers_memory_efficient_attention()

        self.pipeline = AnimationPipeline(vae=controller.vae, text_encoder=controller.text_encoder, tokenizer=controller.tokenizer, unet=controller.unet,
                    scheduler=scheduler_dict["LCM"](**OmegaConf.to_container(controller.inference_config.noise_scheduler_kwargs))).to("cuda")

        if controller.lora_model_state_dict != {}:
            self.pipeline = convert_lora(self.pipeline, controller.lora_model_state_dict, alpha=lora_alpha_slider)

        self.pipeline.unet = convert_lcm_lora(copy.deepcopy(controller.unet), controller.lcm_lora_path, 0.8)
        self.pipeline.to("cuda")
    def predict(
        self,
        prompt: str = Input(default="a cute rabbit, white background, pastel hues, minimal illustration, line art, pen drawing"),
        negative_prompt: str = Input(default="blurry"),
        num_inference_steps: int = Input(default=4),
        guidance_scale: int = Input(default=1),
        width: int = Input(default=512),
        height: int = Input(default=512),
        video_length: int = Input(default=16),
    ) -> Path:
        video_path = inference(self.pipeline, prompt, negative_prompt, num_inference_steps, guidance_scale, width, height, video_length)
        return Path(video_path)