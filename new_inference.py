import os
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from model.pipeline import CatVTONPipeline
from utils import resize_and_crop, resize_and_padding

base_model_path = "runwayml/stable-diffusion-inpainting"
resume_path = "zhengchong/CatVTON"

seed = 42
num_inference_steps = 3
guidance_scale = 2.5

WIDTH = 768
HEIGHT = 1024

pipeline = CatVTONPipeline(
    attn_ckpt_version="mix",
    attn_ckpt=resume_path,
    base_ckpt=base_model_path,
    weight_dtype=torch.float32,
    device="cuda",
    skip_safety_check=True,
)


def preprocess_images(person_image_path, cloth_image_path, mask_image_path):
    vae_processor = VaeImageProcessor(vae_scale_factor=8)
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )

    person_image = Image.open(person_image_path).convert("RGB")
    person_image = resize_and_crop(person_image, (WIDTH, HEIGHT))

    cloth_image = Image.open(cloth_image_path).convert("RGB")
    cloth_image = resize_and_padding(cloth_image, (WIDTH, HEIGHT))

    mask_image = Image.open(mask_image_path).convert("L")
    mask_image = resize_and_crop(mask_image, (WIDTH, HEIGHT))

    preprocessed_person_image = vae_processor.preprocess(person_image, HEIGHT, WIDTH)[0]
    preprocessed_cloth_image = vae_processor.preprocess(cloth_image, HEIGHT, WIDTH)[0]
    preprocessed_mask_image = mask_processor.preprocess(mask_image, HEIGHT, WIDTH)[0]

    return preprocessed_person_image, preprocessed_cloth_image, preprocessed_mask_image


def get_vton(
    person_image_path, cloth_image_path, mask_image_path, cloth_type, username
):

    # 이미지 전처리
    preprocessed_person_image, preprocessed_cloth_image, preprocessed_mask_image = (
        preprocess_images(person_image_path, cloth_image_path, mask_image_path)
    )

    # 난수 고정
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # 결과 생성
    try:
        result = pipeline(
            image=preprocessed_person_image,
            condition_image=preprocessed_cloth_image,
            mask=preprocessed_mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=HEIGHT,
            width=WIDTH,
            generator=generator,
        )[0]
    except:
        print("Error")
        result = pipeline(
            image=preprocessed_person_image[0],
            condition_image=preprocessed_cloth_image[0],
            mask=preprocessed_mask_image[0],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=HEIGHT,
            width=WIDTH,
            generator=generator,
        )[0]

    # 결과 저장 디렉토리 생성
    output_dir = "./vton_output"
    os.makedirs(output_dir, exist_ok=True)

    # 결과 저장
    output_path = os.path.join(
        output_dir, username, f"{username}_{cloth_image_path}_{cloth_type}.jpg"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path)


if __name__ == "__main__":
    get_vton(
        person_image_path="man_test.jpg",
        cloth_image_path="concatenated_image.jpg",
        mask_image_path="man_test_overall.png",
        cloth_type="overall",
        username="wswssds",
    )
