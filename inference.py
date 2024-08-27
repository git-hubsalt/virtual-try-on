from io import BytesIO
import os
import time
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from model.pipeline import CatVTONPipeline
from utils import resize_and_crop, resize_and_padding
import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

# print(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)

s3 = boto3.client(
    "s3",
    region_name="ap-northeast-2",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

SEED = 42
NUM_INFERENCE_STEPS = 50
WIDTH = 768
HEIGHT = 1024

pipeline = CatVTONPipeline(
    attn_ckpt_version="mix",
    attn_ckpt="zhengchong/CatVTON",
    base_ckpt="runwayml/stable-diffusion-inpainting",
    weight_dtype=torch.float32,
    device="cuda",
    skip_safety_check=True,
)


def concat_upper_and_lower(upper_cloth_path, lower_cloth_path):
    # 두 이미지 파일 열기
    image1 = Image.open(upper_cloth_path)
    image2 = Image.open(lower_cloth_path)

    # 두 이미지의 너비를 맞추고, 높이를 합산
    new_width = max(image1.width, image2.width)
    new_height = image1.height + image2.height

    # 새 이미지를 만들고, 이미지1과 이미지2를 붙이기
    new_image = Image.new("RGB", (new_width, new_height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (0, image1.height))

    # # 결과를 저장 또는 보여주기
    # new_image.save("concatenated_image_.jpg")

    return new_image


def preprocess_images(
    person_image_path,
    upper_cloth_path,
    lower_cloth_path,
    mask_image_path,
):
    vae_processor = VaeImageProcessor(vae_scale_factor=8)
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )

    person_image = Image.open(person_image_path).convert("RGB")
    person_image = resize_and_crop(person_image, (WIDTH, HEIGHT))

    # cloth_image = Image.open(cloth_image_path).convert("RGB")
    cloth_image = concat_upper_and_lower(upper_cloth_path, lower_cloth_path)
    cloth_image = resize_and_padding(cloth_image, (WIDTH, HEIGHT))

    mask_image = Image.open(mask_image_path).convert("L")
    mask_image = resize_and_crop(mask_image, (WIDTH, HEIGHT))

    preprocessed_person_image = vae_processor.preprocess(person_image, HEIGHT, WIDTH)[0]
    preprocessed_cloth_image = vae_processor.preprocess(cloth_image, HEIGHT, WIDTH)[0]
    preprocessed_mask_image = mask_processor.preprocess(mask_image, HEIGHT, WIDTH)[0]

    return preprocessed_person_image, preprocessed_cloth_image, preprocessed_mask_image


def save_and_upload_s3(result, username, cloth_type):
    start_timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    # 결과 저장 디렉토리 생성
    output_dir = "./vton_output"
    os.makedirs(output_dir, exist_ok=True)

    # 결과 저장
    output_path = os.path.join(
        output_dir, username, f"{username}_{cloth_type}_{start_timestamp}.jpg"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path)

    # 이미지를 메모리 버퍼에 저장
    buffer = BytesIO()
    result.save(buffer, "JPEG")
    buffer.seek(0)  # 버퍼의 시작 위치로 이동

    bucket_name = "githubsalt-bucket"
    object_name = f"users/{username}/vton_result/{start_timestamp}/result.jpg"

    s3.upload_fileobj(buffer, bucket_name, object_name)


def get_vton(
    person_image_path,
    upper_cloth_path,
    lower_cloth_path,
    mask_image_path,
    cloth_type,
    username,
):
    # 이미지 전처리
    preprocessed_person_image, preprocessed_cloth_image, preprocessed_mask_image = (
        preprocess_images(
            person_image_path,
            upper_cloth_path,
            lower_cloth_path,
            mask_image_path,
        )
    )

    # 난수 고정
    generator = torch.Generator(device="cuda").manual_seed(SEED)

    # 결과 생성
    try:
        result = pipeline(
            image=preprocessed_person_image,
            condition_image=preprocessed_cloth_image,
            mask=preprocessed_mask_image,
            num_inference_steps=NUM_INFERENCE_STEPS,
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
            num_inference_steps=NUM_INFERENCE_STEPS,
            height=HEIGHT,
            width=WIDTH,
            generator=generator,
        )[0]

    # 결과 저장
    save_and_upload_s3(result, username, cloth_type)


if __name__ == "__main__":
    get_vton(
        person_image_path="man_test.jpg",
        upper_cloth_path="upper.jpg",
        lower_cloth_path="lower.jpg",
        mask_image_path="man_test_overall.png",
        cloth_type="overall",
        username="a",
    )
