import os
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

base_model_path = "runwayml/stable-diffusion-inpainting"
resume_path = "zhengchong/CatVTON"

seed = 42
num_inference_steps = 50
guidance_scale = 2.5

WIDTH = 768
HEIGHT = 1024


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

    # 이미지 전처리
    vae_processor = VaeImageProcessor(vae_scale_factor=8)
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )

    person_image = Image.open(person_image).convert("RGB")
    person_image = resize_and_crop(person_image, (WIDTH, HEIGHT))

    cloth_image = Image.open(cloth_image).convert("RGB")
    cloth_image = resize_and_padding(cloth_image, (WIDTH, HEIGHT))

    mask_image = Image.open(mask_image).convert("L")
    mask_image = resize_and_crop(mask_image, (WIDTH, HEIGHT))

    preprocessed_person_image = vae_processor.preprocess(person_image, HEIGHT, WIDTH)[0]
    preprocessed_cloth_image = vae_processor.preprocess(cloth_image, HEIGHT, WIDTH)[0]
    preprocessed_mask_image = mask_processor.preprocess(mask_image, HEIGHT, WIDTH)[0]

    # 난수 고정
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # 결과 생성
    try:
        results = pipeline(
            image=preprocessed_person_image,
            condition_image=preprocessed_cloth_image,
            mask=preprocessed_mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=HEIGHT,
            width=WIDTH,
            generator=generator,
        )
    except:
        print("Error")
        results = pipeline(
            image=preprocessed_person_image[0],
            condition_image=preprocessed_cloth_image[0],
            mask=preprocessed_mask_image[0],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=HEIGHT,
            width=WIDTH,
            generator=generator,
        )

    # 결과 저장 디렉토리 생성
    output_dir = "./vton_output"
    os.makedirs(output_dir, exist_ok=True)

    # 결과 저장
    output_path = os.path.join(output_dir, username, f"{username}_{cloth_type}.jpg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results[0].save(output_path)


if __name__ == "__main__":
    main(
        person_image="man_test.jpg",
        cloth_image="concatenated_image.jpg",
        mask_image="man_test_overall.png",
        cloth_type="overall",
        username="wwssds",
    )
