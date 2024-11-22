# omoib-virtual try on
- 사용자가 추천받은 코디나 옷장의 옷을 가상으로 입어보는 기능
- 실제로 옷을 입어보지 않고도, 코디가 자신에게 어울리는지 미리 확인 가능
<br>

## Process
1️⃣ 회원가입 시 등록한 user image에서 상의+하의를 masking한 image 생성<br>
2️⃣ 상의와 하의를 모두 피팅해야하는 경우, 한번에 수행하기 위해 item image를 concat<br>
3️⃣ user image, masking image, item image를 CatVTON 모델에 넣어 피팅된 이미지 반환
<br>
<br>

## Inference 결과
<img src="https://github.com/user-attachments/assets/0b215759-26a8-4390-8c0f-e347ca95fb2a" width="800" height="400"/>
