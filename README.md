# omoib-virtual try on
**AI 패션 플랫폼, 오늘 모 입지? 오모입!**
<br>
<br>
<br>

## 가상 피팅
- 사용자가 추천받은 코디나 옷장의 옷을 가상으로 입어보는 기능
- 실제로 옷을 입어보지 않고도, 코디가 자신에게 어울리는지 미리 확인 가능
<br>

## 활용 모델: CatVTON
[CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models](https://github.com/Zheng-Chong/CatVTON)
<img src="https://github.com/user-attachments/assets/d35efb07-a6f6-4939-8d4e-726bc46ac98c" width="900" height="370"/>

- Diffusion 기반의 Image Virtual-Try On 모델
- condition을 처리하기 위한 추가적인 구조 없이 origininal diffusion 모듈만 사용
- 불필요한 condition과 전처리 단계를 제거하여 person, cloth, cloth-agnostic mask만을 필요로 하는 간소화된 inference 단계
<br>

## Process
1️⃣ 회원가입 시 등록한 user body image에서 densepose와 SCHP를 이용해 상의+하의를 masking한 image 생성<br>
2️⃣ 상의와 하의를 모두 피팅해야하는 경우, 한번에 수행하기 위해 item image를 concat<br>
3️⃣ user body image, masking image, item image를 CatVTON 모델에 넣어 피팅된 이미지 반환
