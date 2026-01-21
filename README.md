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

## 피드백 반영

1. 브랜치명 변경 - (어떤 업무인지 확인 가능)
2. 폴더 구조 변경 (task1, task2, ...)
3. 2차 업무부터 src, data, test 로 나누어서 진행하겠습니다.

- **2차 세션 후 최종 변경사항 master로 병합 예정**

---

# 2차 업무
## 업무 진행 방식
- 튜토리얼 참고 및 학습
- Jupyter Notebook 실습
- 최종 코드 .py파일 생성
## [2D-3D 변환 알고리즘] - 그레이 스케일 기반
이번 실습 이미지로는 테이블에 포크,나이프, 접시, 음식이 놓여있는 이미지를 이용하였습니다.
그레이 스케일 변환 후 depth_map을 생성하여 생성된 depth_map으로 3D_포인트 클라우드를 생성합니다.

- **진행 요약 : 그레이 스케일 변환 → Depth_Map 생성 → 3D Point Cloud 생성**
<img width="1131" height="359" alt="image" src="https://github.com/user-attachments/assets/fa9b18a2-8f44-4ea4-898b-53801f44efca" />

## [현재 알고리즘의 문제점] - 그레이 스케일 기반
그레이 스케일 기반으로 생성한 depth_map은 하얀색 영역을 기반으로 생성하므로 주로 빛의 밝기로 입체 형상을 파악합니다.

하지만, 그 외의 Edge Case를 고려해야 합니다. → ex) 하얀색 접시, 빛 반사로 인한 일부 현상, 그 외의 하얀색을 띄는 물체
<img width="1287" height="515" alt="image" src="https://github.com/user-attachments/assets/f673ff7d-aefc-408b-9728-ca41d90627e2" />

## [Unit - Test]
5개의 테스트 함수를 작성하였으며, 정상 작동, 예외 처리, 동작 검증을 하는 테스트 코드로 작성하였습니다.
실제 3D 매핑 로직 적용 중 데이터 타입 변환 오류를 발견하여 수정하였습니다.
<img width="986" height="315" alt="image" src="https://github.com/user-attachments/assets/60bd14ce-f7e3-4614-a1f6-e62e3b8d31ec" />

Unit Test 코드를 작성하여 실행해 본 결과 3D 포인트의 데이터 타입이 기존 np.float32에서 달라지는 점을 확인하여 수정하여 코드를 완성하였습니다.
<img width="1300" height="219" alt="image" src="https://github.com/user-attachments/assets/9744df23-dbdb-41b1-a3b5-575f807fd17a" />
