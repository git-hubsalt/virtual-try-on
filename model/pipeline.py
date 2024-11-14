import inspect
import os
from typing import Union

import PIL
import numpy as np
import torch
import tqdm
from accelerate import load_checkpoint_in_model
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import snapshot_download
from transformers import CLIPImageProcessor

from model.attn_processor import SkipAttnProcessor
from model.utils import get_trainable_module, init_adapter
from utils import (
    compute_vae_encodings,
    numpy_to_pil,
    prepare_image,
    prepare_mask_image,
    resize_and_crop,
    resize_and_padding,
)


class CatVTONPipeline:
    def __init__(
        self,
        base_ckpt,
        attn_ckpt,
        attn_ckpt_version="mix",
        weight_dtype=torch.float32,
        device="cuda",
        compile=False,
        skip_safety_check=False,
        use_tf32=False,
    ):
        self.device = device
        self.weight_dtype = weight_dtype
        self.skip_safety_check = skip_safety_check
        print("device: " + self.device)
        
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(
            device, dtype=weight_dtype
        )

        if not skip_safety_check:
            self.feature_extractor = CLIPImageProcessor.from_pretrained(
                base_ckpt, subfolder="feature_extractor"
            )
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                base_ckpt, subfolder="safety_checker"
            ).to(device, dtype=weight_dtype)

        self.noise_scheduler = DDIMScheduler.from_pretrained(
            base_ckpt, subfolder="scheduler"
        )
        
        self.unet = UNet2DConditionModel.from_pretrained(
            base_ckpt, subfolder="unet"
        ).to(device, dtype=weight_dtype)
        init_adapter(
            self.unet, cross_attn_cls=SkipAttnProcessor
        )  # Skip Cross-Attention
        self.attn_modules = get_trainable_module(self.unet, "attention")
        self.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version)
        # Pytorch 2.0 Compile
        if compile:
            self.unet = torch.compile(self.unet)
            self.vae = torch.compile(self.vae, mode="reduce-overhead")

        # Enable TF32 for faster training on Ampere GPUs (A100 and RTX 30 series).
        if use_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

    def auto_attn_ckpt_load(self, attn_ckpt, version):
        sub_folder = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }[version]
        if os.path.exists(attn_ckpt):
            load_checkpoint_in_model(
                self.attn_modules, os.path.join(attn_ckpt, sub_folder, "attention")
            )
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            print(f"Downloaded {attn_ckpt} to {repo_path}")
            load_checkpoint_in_model(
                self.attn_modules, os.path.join(repo_path, sub_folder, "attention")
            )

    def run_safety_checker(self, image):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            safety_checker_input = self.feature_extractor(
                image, return_tensors="pt"
            ).to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.to(self.weight_dtype),
            )
        return image, has_nsfw_concept

    def check_inputs(self, image, condition_image, mask, width, height):
        if (
            isinstance(image, torch.Tensor)
            and isinstance(condition_image, torch.Tensor)
            and isinstance(mask, torch.Tensor)
        ):
            return image, condition_image, mask
        assert image.size == mask.size, "Image and mask must have the same size"
        image = resize_and_crop(image, (width, height))
        mask = resize_and_crop(mask, (width, height))
        condition_image = resize_and_padding(condition_image, (width, height))
        return image, condition_image, mask

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self,
        image: Union[PIL.Image.Image, torch.Tensor],
        condition_image: Union[PIL.Image.Image, torch.Tensor],
        mask: Union[PIL.Image.Image, torch.Tensor],
        num_inference_steps: int = 50,
        guidance_scale: float = 2.5,
        height: int = 1024,
        width: int = 768,
        generator=None,
        eta=1.0,
        **kwargs,
    ):
        concat_dim = -2  # FIXME: y axis concat
        # Prepare inputs to Tensor
        image, condition_image, mask = self.check_inputs(
            image, condition_image, mask, width, height
        )
        image = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(condition_image).to(
            self.device, dtype=self.weight_dtype
        )
        mask = prepare_mask_image(mask).to(self.device, dtype=self.weight_dtype)
        # Mask image
        masked_image = image * (mask < 0.5)
        # VAE encoding
        masked_latent = compute_vae_encodings(masked_image, self.vae)
        condition_latent = compute_vae_encodings(condition_image, self.vae)
        mask_latent = torch.nn.functional.interpolate(
            mask, size=masked_latent.shape[-2:], mode="nearest"
        )
        del image, mask, condition_image
        # Concatenate latents
        masked_latent_concat = torch.cat(
            [masked_latent, condition_latent], dim=concat_dim
        )
        mask_latent_concat = torch.cat(
            [mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim
        )
        # Prepare noise
        latents = randn_tensor(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=self.weight_dtype,
        )
        # Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        latents = latents * self.noise_scheduler.init_noise_sigma
        # Classifier-Free Guidance
        if do_classifier_free_guidance := (guidance_scale > 1.0):
            masked_latent_concat = torch.cat(
                [
                    torch.cat(
                        [masked_latent, torch.zeros_like(condition_latent)],
                        dim=concat_dim,
                    ),
                    masked_latent_concat,
                ]
            )
            mask_latent_concat = torch.cat([mask_latent_concat] * 2)

        # Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.noise_scheduler.order
        )
        with tqdm.tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                non_inpainting_latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                non_inpainting_latent_model_input = (
                    self.noise_scheduler.scale_model_input(
                        non_inpainting_latent_model_input, t
                    )
                )
                # prepare the input for the inpainting model
                inpainting_latent_model_input = torch.cat(
                    [
                        non_inpainting_latent_model_input,
                        mask_latent_concat,
                        masked_latent_concat,
                    ],
                    dim=1,
                )
                # predict the noise residual
                noise_pred = self.unet(
                    inpainting_latent_model_input,
                    t.to(self.device),
                    encoder_hidden_states=None,  # FIXME
                    return_dict=False,
                )[0]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.noise_scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.noise_scheduler.order == 0
                ):
                    progress_bar.update()

        # Decode the final latents
        latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.device, dtype=self.weight_dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = numpy_to_pil(image)

        # Safety Check
        if not self.skip_safety_check:
            current_script_directory = os.path.dirname(os.path.realpath(__file__))
            nsfw_image = os.path.join(
                os.path.dirname(current_script_directory), "resource", "img", "NSFW.jpg"
            )
            nsfw_image = PIL.Image.open(nsfw_image).resize(image[0].size)
            image_np = np.array(image[0])
            _, has_nsfw_concept = self.run_safety_checker(image=image_np)
            for i, not_safe in enumerate(has_nsfw_concept):
                if not_safe:
                    image[i] = nsfw_image
        return image
