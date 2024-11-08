import requests

response = requests.head("https://huggingface.co")

# 헤더 정보 출력
for header, value in response.headers.items():
    print(f"{header}: {value}")
