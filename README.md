# Comento_Computer_Vision
Comento Computer Vision Online BootCamp

# 1차 업무
## 업무 진행 방식
- 튜토리얼 참고 및 학습
- Jupyter Notebook 실습
- 최종 코드 .py파일 생성
## [기본 업무]
실습 이미지로는 다양한 색상이 포함된 풍선 이미지를 사용하였습니다.
기존 코드의 마스킹 결과를 확인 후 좀 더 유연한 빨간색 영역 검출을 위해 lower_red 값을 조정하였으며, 그 결과 오른쪽 이미지와 같이 빨간색 풍선 영역이 보다 명확하게 검출되는 것을 확인하였습니다.

<img width="1528" height="464" alt="image" src="https://github.com/user-attachments/assets/326dba8b-9746-4433-8e41-60ff4f2c808c" />

## [추가 요청(기본 문제)]
이미지는 허깅페이스에서 데이터셋을 이용하였습니다. ethz/food 데이터셋이며, 20장의 이미지만을 로드하였습니다.
https://huggingface.co/datasets/ethz/food101

코드 실습 및 진행 현황 확인을 위해 Jupyter notebook 환경에서 진행하였습니다.

[기본 문제]와 [심화 문제]를 진행하였으며, 마지막 .py파일 에서는 [심화 문제]로 필터링 과정을 거치고 [기본 문제] 전처리 코드를 적용하는
방식으로 코드를 구성하였습니다.

- **진행 요약 : Jupyter Notebook 실습 → [기본 문제] → [심화 문제] → .py파일로 최종 반영**
<img width="1982" height="607" alt="image" src="https://github.com/user-attachments/assets/65240d85-c3d5-4de0-80c7-71d5a61f9da3" />

## [추가 요청(심화 문제)]
[심화 문제]는 먼저 20가지 이미지 파일을 출력하여 확인 후 진행하였습니다.

어두운 이미지 필터링 같은 경우 각 이미지를 그레이 스케일로 변환 후 다음과 같은 방식으로 분류하였습니다.
- **(0 ~ 1)정규화 → 평균계산 → 임계값 이하의 이미지 필터링**
<img width="1578" height="70" alt="image" src="https://github.com/user-attachments/assets/02c7b1da-846e-47d2-abd5-edc9f5ba3b34" />

<img width="1769" height="651" alt="image" src="https://github.com/user-attachments/assets/a30b54bc-e793-4d80-8aa8-12787ab8b5d1" />

---

# 2차 업무
