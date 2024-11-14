from io import BytesIO
import json
import os
import requests
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from model.pipeline import CatVTONPipeline
from utils import resize_and_crop, resize_and_padding
import boto3
from dotenv import load_dotenv

load_dotenv()

print("--------------------------------")
print("is cuda? : ", torch.cuda.is_available())

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
NUM_STEP = int(os.environ.get("NUM_STEP", 15))
QUEUE_URL = "https://sqs.ap-northeast-2.amazonaws.com/565393031158/omoib-vton-queue"

print(f"NUM_STEP should be : {NUM_STEP}")

s3 = boto3.client(
    "s3",
    region_name="ap-northeast-2",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

sqs = boto3.client(
    "sqs",
    region_name="ap-northeast-2",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


SEED = 42
NUM_INFERENCE_STEPS = NUM_STEP
WIDTH = 768
HEIGHT = 1024

pipeline = CatVTONPipeline(
    attn_ckpt_version="mix",
    attn_ckpt="zhengchong/CatVTON",
    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
    weight_dtype=torch.float16,
    device="cuda",
    skip_safety_check=True,
)


def concat_upper_and_lower(image1, image2):
    # 두 이미지의 너비를 맞추고, 높이를 합산
    new_width = max(image1.width, image2.width)
    new_height = image1.height + image2.height

    # 새 이미지를 만들고, 이미지1과 이미지2를 붙이기
    new_image = Image.new("RGB", (new_width, new_height))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (0, image1.height))

    return new_image


def preprocess_images(
        person_image_url,
        upper_cloth_url,
        lower_cloth_url,
        mask_image_url,
    ):
    response = requests.get(person_image_url)
    if response.status_code == 200:
        person_image = Image.open(BytesIO(response.content))

    response = requests.get(upper_cloth_url)
    if response.status_code == 200:
        upper_cloth_image = Image.open(BytesIO(response.content))

    response = requests.get(lower_cloth_url)
    if response.status_code == 200:
        lower_cloth_image = Image.open(BytesIO(response.content))

    response = requests.get(mask_image_url)
    if response.status_code == 200:
        mask_image = Image.open(BytesIO(response.content))

    vae_processor = VaeImageProcessor(vae_scale_factor=8)
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    )

    person_image = resize_and_crop(person_image, (WIDTH, HEIGHT))

    cloth_image = concat_upper_and_lower(upper_cloth_image, lower_cloth_image)
    cloth_image = resize_and_padding(cloth_image, (WIDTH, HEIGHT))

    mask_image = resize_and_crop(mask_image, (WIDTH, HEIGHT))

    preprocessed_person_image = vae_processor.preprocess(person_image, HEIGHT, WIDTH)[0]
    preprocessed_cloth_image = vae_processor.preprocess(cloth_image, HEIGHT, WIDTH)[0]
    preprocessed_mask_image = mask_processor.preprocess(mask_image, HEIGHT, WIDTH)[0]

    return preprocessed_person_image, preprocessed_cloth_image, preprocessed_mask_image


def save_and_upload_s3(result, username, cloth_type, timestamp):
    # 결과 저장 디렉토리 생성
    output_dir = "./vton_output"
    os.makedirs(output_dir, exist_ok=True)

    # 결과 저장
    output_path = os.path.join(
        output_dir, username, f"{username}_{cloth_type}_{timestamp}.jpg"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path)

    # 이미지를 메모리 버퍼에 저장
    buffer = BytesIO()
    result.save(buffer, "JPEG")
    buffer.seek(0)  # 버퍼의 시작 위치로 이동

    bucket_name = "githubsalt-bucket"
    object_name = f"users/{username}/vton_result/{timestamp}/result.jpg"

    s3.upload_fileobj(buffer, bucket_name, object_name)


def send_sqs(username, timestamp):
    message_body = {
        "userId": username,
        "initial_timestamp": timestamp,
    }

    try:
        response = sqs.send_message(
            QueueUrl=QUEUE_URL, MessageBody=json.dumps(message_body)
        )
        print(f"Message sent to SQS with MessageId: {response['MessageId']}")

        return {"statusCode": 200, "body": json.dumps("Message sent successfully!")}
    except Exception as e:
        print(f"Error sending message: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps("Error sending message to SQS"),
        }

def get_vton(
    person_image_url,
    upper_cloth_url,
    lower_cloth_url,
    mask_image_url,
    cloth_type,
    username,
    timestamp,
):
    # 이미지 전처리
    preprocessed_person_image, preprocessed_cloth_image, preprocessed_mask_image = (
        preprocess_images(
            person_image_url,
            upper_cloth_url,
            lower_cloth_url,
            mask_image_url,
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
    save_and_upload_s3(result, username, cloth_type, timestamp)
    send_sqs(username, timestamp)


if __name__ == "__main__":

    person_image_url = "https://githubsalt-bucket.s3.ap-northeast-2.amazonaws.com/users/a/row/240828-000129/row.jpg?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEP3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDmFwLW5vcnRoZWFzdC0yIkcwRQIhAIJxXgd0nQ%2FXy3ZiwFcYILynyRONs65lVydsBlTYVEYfAiAVt13dH3vKJOVpiRQs203VSzL3PWTL%2BF1ntV5Yd0Vt4yroAggnEAAaDDU2NTM5MzAzMTE1OCIM8TugZcgcWaPKXSOuKsUC%2FamT3yOZN%2FjZI58Uu%2FPZHYpuskDnRGfkcDEcHi77wDxNWE1R8DwY5sHsPq9nc1vWtAkXvlYN75poixDJfMPHuslSAFk0zHxZvoN2DqGaMF4E7kud4eEdoDJFDrtNhQgYKD9P%2BK6c5MkPk28jGOIiOne2gt2Ey7ipevPQzXS9po9sbY2jptr5zM19TEVBbTEzeCquF%2BE5rGJ%2BgTn3NwYq6B5P8gqst%2BTd2QQpapEfzWxpTz9B4x5LoOH0FJ2avsqLqnTLC3V0TQv510rcwNQ%2FOp7JM2N%2Fq43RzgqKApCAAKkyWn0S7azWWpSsapONPLOFZK7%2FRa4nZJv9s3L9CsvXzjoSELdEZha7MRFm%2BhTIkoW5qMDomBKi2Nq8ZX%2FMe7se%2FiKGts31%2FBWkGSZFJdg%2B5X7IxSjuAjEr7WD0yoVAMmCZ1Q6CXjCXtu%2B2BjqzAlZHhHBi6YHjFysGk6fvl4U5C41mb4A9JqYuOaN%2FfC%2F06YKGoXznhgUJQIPgwQ7oz%2FnUHAvE2M04oW3G%2F4gscz5FzTU6z0zdhQrYbuOYAFtVliCFc7%2FhjZOdmYayOJ5wSg4TsMXbt1eMduN9C0OyiGQEoHNR06HOux%2BU801joOioJXZIt2VQ3iSVcryFAVS1cWr1IMuWtfrTS%2FZ7H2nPoFpNAWhoT9%2FpqGrWyXQ2KqmHe%2BKbp%2FHnJ2K0A3%2F8tFd3LlUZHs%2B7yO7D85lOG%2Bp1uNSb3habpp9MoKvm%2BBmixajhEv096T%2F67oziomQ%2Bn7SRm%2FT%2F4yhiaMi7W6St4eDtU9XPXh2nkTOhpLUVGNcOOL858PCoSDjttZxXrqX2eGSJ4EbQYQjTvmcg4ZcxylJLRB8D3MI%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240907T052450Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAYHJAM773DFKNUQLW%2F20240907%2Fap-northeast-2%2Fs3%2Faws4_request&X-Amz-Signature=524559e2fcd1a2402a16d4b1f2062bf2aa88163d068a1c8298986b7f68904070"

    upper_cloth_url = "https://githubsalt-bucket.s3.ap-northeast-2.amazonaws.com/users/a/items/closet/upper.jpg?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEP3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDmFwLW5vcnRoZWFzdC0yIkcwRQIhAIJxXgd0nQ%2FXy3ZiwFcYILynyRONs65lVydsBlTYVEYfAiAVt13dH3vKJOVpiRQs203VSzL3PWTL%2BF1ntV5Yd0Vt4yroAggnEAAaDDU2NTM5MzAzMTE1OCIM8TugZcgcWaPKXSOuKsUC%2FamT3yOZN%2FjZI58Uu%2FPZHYpuskDnRGfkcDEcHi77wDxNWE1R8DwY5sHsPq9nc1vWtAkXvlYN75poixDJfMPHuslSAFk0zHxZvoN2DqGaMF4E7kud4eEdoDJFDrtNhQgYKD9P%2BK6c5MkPk28jGOIiOne2gt2Ey7ipevPQzXS9po9sbY2jptr5zM19TEVBbTEzeCquF%2BE5rGJ%2BgTn3NwYq6B5P8gqst%2BTd2QQpapEfzWxpTz9B4x5LoOH0FJ2avsqLqnTLC3V0TQv510rcwNQ%2FOp7JM2N%2Fq43RzgqKApCAAKkyWn0S7azWWpSsapONPLOFZK7%2FRa4nZJv9s3L9CsvXzjoSELdEZha7MRFm%2BhTIkoW5qMDomBKi2Nq8ZX%2FMe7se%2FiKGts31%2FBWkGSZFJdg%2B5X7IxSjuAjEr7WD0yoVAMmCZ1Q6CXjCXtu%2B2BjqzAlZHhHBi6YHjFysGk6fvl4U5C41mb4A9JqYuOaN%2FfC%2F06YKGoXznhgUJQIPgwQ7oz%2FnUHAvE2M04oW3G%2F4gscz5FzTU6z0zdhQrYbuOYAFtVliCFc7%2FhjZOdmYayOJ5wSg4TsMXbt1eMduN9C0OyiGQEoHNR06HOux%2BU801joOioJXZIt2VQ3iSVcryFAVS1cWr1IMuWtfrTS%2FZ7H2nPoFpNAWhoT9%2FpqGrWyXQ2KqmHe%2BKbp%2FHnJ2K0A3%2F8tFd3LlUZHs%2B7yO7D85lOG%2Bp1uNSb3habpp9MoKvm%2BBmixajhEv096T%2F67oziomQ%2Bn7SRm%2FT%2F4yhiaMi7W6St4eDtU9XPXh2nkTOhpLUVGNcOOL858PCoSDjttZxXrqX2eGSJ4EbQYQjTvmcg4ZcxylJLRB8D3MI%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240907T052946Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAYHJAM773DFKNUQLW%2F20240907%2Fap-northeast-2%2Fs3%2Faws4_request&X-Amz-Signature=d2408ad2a4e2801ef270c62d3c198326cb7c19596d3974e4b0f8837508c5f421"

    lower_cloth_url = "https://githubsalt-bucket.s3.ap-northeast-2.amazonaws.com/users/a/items/closet/lower.jpg?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEP3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDmFwLW5vcnRoZWFzdC0yIkcwRQIhAIJxXgd0nQ%2FXy3ZiwFcYILynyRONs65lVydsBlTYVEYfAiAVt13dH3vKJOVpiRQs203VSzL3PWTL%2BF1ntV5Yd0Vt4yroAggnEAAaDDU2NTM5MzAzMTE1OCIM8TugZcgcWaPKXSOuKsUC%2FamT3yOZN%2FjZI58Uu%2FPZHYpuskDnRGfkcDEcHi77wDxNWE1R8DwY5sHsPq9nc1vWtAkXvlYN75poixDJfMPHuslSAFk0zHxZvoN2DqGaMF4E7kud4eEdoDJFDrtNhQgYKD9P%2BK6c5MkPk28jGOIiOne2gt2Ey7ipevPQzXS9po9sbY2jptr5zM19TEVBbTEzeCquF%2BE5rGJ%2BgTn3NwYq6B5P8gqst%2BTd2QQpapEfzWxpTz9B4x5LoOH0FJ2avsqLqnTLC3V0TQv510rcwNQ%2FOp7JM2N%2Fq43RzgqKApCAAKkyWn0S7azWWpSsapONPLOFZK7%2FRa4nZJv9s3L9CsvXzjoSELdEZha7MRFm%2BhTIkoW5qMDomBKi2Nq8ZX%2FMe7se%2FiKGts31%2FBWkGSZFJdg%2B5X7IxSjuAjEr7WD0yoVAMmCZ1Q6CXjCXtu%2B2BjqzAlZHhHBi6YHjFysGk6fvl4U5C41mb4A9JqYuOaN%2FfC%2F06YKGoXznhgUJQIPgwQ7oz%2FnUHAvE2M04oW3G%2F4gscz5FzTU6z0zdhQrYbuOYAFtVliCFc7%2FhjZOdmYayOJ5wSg4TsMXbt1eMduN9C0OyiGQEoHNR06HOux%2BU801joOioJXZIt2VQ3iSVcryFAVS1cWr1IMuWtfrTS%2FZ7H2nPoFpNAWhoT9%2FpqGrWyXQ2KqmHe%2BKbp%2FHnJ2K0A3%2F8tFd3LlUZHs%2B7yO7D85lOG%2Bp1uNSb3habpp9MoKvm%2BBmixajhEv096T%2F67oziomQ%2Bn7SRm%2FT%2F4yhiaMi7W6St4eDtU9XPXh2nkTOhpLUVGNcOOL858PCoSDjttZxXrqX2eGSJ4EbQYQjTvmcg4ZcxylJLRB8D3MI%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240907T053009Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAYHJAM773DFKNUQLW%2F20240907%2Fap-northeast-2%2Fs3%2Faws4_request&X-Amz-Signature=b231c4bc3485d94da4e1b7bba013968c40b77907bffc477bc006a02918617410"

    mask_image_url = "https://githubsalt-bucket.s3.ap-northeast-2.amazonaws.com/users/a/masking/240828-011010/overall.png?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEP3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDmFwLW5vcnRoZWFzdC0yIkcwRQIhAIJxXgd0nQ%2FXy3ZiwFcYILynyRONs65lVydsBlTYVEYfAiAVt13dH3vKJOVpiRQs203VSzL3PWTL%2BF1ntV5Yd0Vt4yroAggnEAAaDDU2NTM5MzAzMTE1OCIM8TugZcgcWaPKXSOuKsUC%2FamT3yOZN%2FjZI58Uu%2FPZHYpuskDnRGfkcDEcHi77wDxNWE1R8DwY5sHsPq9nc1vWtAkXvlYN75poixDJfMPHuslSAFk0zHxZvoN2DqGaMF4E7kud4eEdoDJFDrtNhQgYKD9P%2BK6c5MkPk28jGOIiOne2gt2Ey7ipevPQzXS9po9sbY2jptr5zM19TEVBbTEzeCquF%2BE5rGJ%2BgTn3NwYq6B5P8gqst%2BTd2QQpapEfzWxpTz9B4x5LoOH0FJ2avsqLqnTLC3V0TQv510rcwNQ%2FOp7JM2N%2Fq43RzgqKApCAAKkyWn0S7azWWpSsapONPLOFZK7%2FRa4nZJv9s3L9CsvXzjoSELdEZha7MRFm%2BhTIkoW5qMDomBKi2Nq8ZX%2FMe7se%2FiKGts31%2FBWkGSZFJdg%2B5X7IxSjuAjEr7WD0yoVAMmCZ1Q6CXjCXtu%2B2BjqzAlZHhHBi6YHjFysGk6fvl4U5C41mb4A9JqYuOaN%2FfC%2F06YKGoXznhgUJQIPgwQ7oz%2FnUHAvE2M04oW3G%2F4gscz5FzTU6z0zdhQrYbuOYAFtVliCFc7%2FhjZOdmYayOJ5wSg4TsMXbt1eMduN9C0OyiGQEoHNR06HOux%2BU801joOioJXZIt2VQ3iSVcryFAVS1cWr1IMuWtfrTS%2FZ7H2nPoFpNAWhoT9%2FpqGrWyXQ2KqmHe%2BKbp%2FHnJ2K0A3%2F8tFd3LlUZHs%2B7yO7D85lOG%2Bp1uNSb3habpp9MoKvm%2BBmixajhEv096T%2F67oziomQ%2Bn7SRm%2FT%2F4yhiaMi7W6St4eDtU9XPXh2nkTOhpLUVGNcOOL858PCoSDjttZxXrqX2eGSJ4EbQYQjTvmcg4ZcxylJLRB8D3MI%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240907T053036Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAYHJAM773DFKNUQLW%2F20240907%2Fap-northeast-2%2Fs3%2Faws4_request&X-Amz-Signature=a0c8adc7b087cec834fd24c5b0d85c217268a622bdfb029006bc8c09f11d2fd5"

    get_vton(
        person_image_url=person_image_url,
        upper_cloth_url=upper_cloth_url,
        lower_cloth_url=lower_cloth_url,
        mask_image_url=mask_image_url,
        cloth_type="overall",
        username="rtr",
        timestamp="20210901-000000",
    )
