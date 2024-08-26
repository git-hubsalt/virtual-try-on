import os
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from model.pipeline import CatVTONPipeline

base_model_path = "runwayml/stable-diffusion-inpainting"
resume_path = "zhengchong/CatVTON"

seed = 555
num_inference_steps = 10
guidance_scale = 2.5

width = 768
height = 1024


def main(person_image, cloth_image, mask_image, cloth_type, username):

    # Pipeline
    pipeline = CatVTONPipeline(
        attn_ckpt_version="mix",
        attn_ckpt=resume_path,
        base_ckpt=base_model_path,
        weight_dtype=torch.float32,
        device="cuda",
        skip_safety_check=True,
    )

    vae_processor = VaeImageProcessor(vae_scale_factor=8)
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )

    person_image = Image.open(person_image)
    cloth_image = Image.open(cloth_image)
    mask_image = Image.open(mask_image)

    preprocessed_person_image = (
        vae_processor.preprocess(person_image, height, width)[0],
    )
    preprocessed_cloth_image = (
        vae_processor.preprocess(cloth_image, height, width)[0],
    )
    preprocessed_mask_image = mask_processor.preprocess(mask_image, height, width)[0]

    # 난수 고정
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # 결과 저장 디렉토리 생성
    output_dir = "./vton_output"
    os.makedirs(output_dir, exist_ok=True)

    results = pipeline(
        preprocessed_person_image[0],
        preprocessed_cloth_image[0],
        preprocessed_mask_image[0],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    )

    print("this is results: ", results)

    output_path = os.path.join(output_dir, username, f"{username}_{cloth_type}.jpg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results[0].save(output_path)


if __name__ == "__main__":
    main(
        person_image="man_test.jpg",
        cloth_image="upper.jpg",
        mask_image="man_test_mask.png",
        cloth_type="upper",
        username="jongmin",
    )
