# AICE Associate 모의고사 🤖

**AICE Associate 실기 대비 모의고사 4회분 + 왕기초 개념노트**
정답과 해설이 들어 있는 Jupyter 노트북이에요. 연습 데이터는 노트북 안에서 자동으로 만들어져서 **다운로드 없이 Colab에서 바로** 풀 수 있어요.

## 📚 노트북 목록

| 노트북 | 내용 | 바로 열기 |
|---|---|---|
| [왕기초 개념노트](00_AICE_왕기초_개념노트.ipynb) | 처음이라면 여기부터. pandas, 전처리, 분류와 회귀, 딥러닝 용어 정리 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/heejung72/AICE-Associate-mock-exam/blob/main/00_AICE_%EC%99%95%EA%B8%B0%EC%B4%88_%EA%B0%9C%EB%85%90%EB%85%B8%ED%8A%B8.ipynb) |
| [01회 이진 분류](01_AICE_모의고사_이진분류_고객이탈.ipynb) | 통신사 고객 이탈 예측. 기본 흐름 익히기 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/heejung72/AICE-Associate-mock-exam/blob/main/01_AICE_%EB%AA%A8%EC%9D%98%EA%B3%A0%EC%82%AC_%EC%9D%B4%EC%A7%84%EB%B6%84%EB%A5%98_%EA%B3%A0%EA%B0%9D%EC%9D%B4%ED%83%88.ipynb) |
| [02회 회귀](02_AICE_모의고사_회귀_배달시간.ipynb) | 배달 소요시간 예측. json, map, MinMaxScaler, MAE와 R2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/heejung72/AICE-Associate-mock-exam/blob/main/02_AICE_%EB%AA%A8%EC%9D%98%EA%B3%A0%EC%82%AC_%ED%9A%8C%EA%B7%80_%EB%B0%B0%EB%8B%AC%EC%8B%9C%EA%B0%84.ipynb) |
| [03회 다중 분류](03_AICE_모의고사_다중분류_공정불량.ipynb) | 공정 불량 등급 예측. 중복 제거, 자료형 변환, LabelEncoder, softmax | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/heejung72/AICE-Associate-mock-exam/blob/main/03_AICE_%EB%AA%A8%EC%9D%98%EA%B3%A0%EC%82%AC_%EB%8B%A4%EC%A4%91%EB%B6%84%EB%A5%98_%EA%B3%B5%EC%A0%95%EB%B6%88%EB%9F%89.ipynb) |
| [04회 종합 (실전)](04_AICE_모의고사_종합_대출연체.ipynb) | 대출 연체 예측. IQR, LogisticRegression, f1-score. 60분 타이머 추천 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/heejung72/AICE-Associate-mock-exam/blob/main/04_AICE_%EB%AA%A8%EC%9D%98%EA%B3%A0%EC%82%AC_%EC%A2%85%ED%95%A9_%EB%8C%80%EC%B6%9C%EC%97%B0%EC%B2%B4.ipynb) |

## 🧭 추천 공부 순서 (3일 완성)

| 일차 | 할 일 |
|---|---|
| 1일차 | 00 개념노트 읽기 → 01회를 정답 보면서 따라 치기 |
| 2일차 | 02회 (회귀), 03회 (다중 분류) 풀기 |
| 3일차 | 04회를 **정답 안 보고 60분 안에** 풀기 → 틀린 유형만 복습 |

## ✍️ 푸는 방법

1. 위 표에서 **Open in Colab** 클릭 (또는 다운로드해서 Jupyter로 열기)
2. 맨 위 **`0. 준비` 셀 실행** → 연습용 데이터 파일 생성
3. 문제 읽고 빈 셀에 직접 코드 작성
4. 막히면 **👉 정답 보기** 클릭, 해설 읽고 따라 치기

## 🗺️ 시험 흐름 (14문항 거의 이 순서)

```
라이브러리 import → 데이터 읽기 → 시각화 → 컬럼 삭제 → 결측치 → 이상치
→ 인코딩 → train_test_split → 스케일링 → 머신러닝 → 평가 → 딥러닝 → 학습 곡선
```

## 🔑 꼭 외울 것

| 문제 유형 | 출력층 | loss |
|---|---|---|
| 이진 분류 | `Dense(1, activation='sigmoid')` | `binary_crossentropy` |
| 다중 분류 | `Dense(클래스수, activation='softmax')` | `sparse_categorical_crossentropy` |
| 회귀 | `Dense(1)` | `mse` |

- `train_test_split` 반환 순서: `X_train, X_test, y_train, y_test`
- test 데이터는 `transform`만 (fit 금지)
- 분류는 `stratify=y`, 회귀는 stratify 없이

## 🛠️ 환경

Python 3, pandas, numpy, scikit-learn, seaborn, matplotlib, tensorflow (Colab 기본 환경에서 바로 실행돼요)

## ⚠️ 안내

공식 기출문제가 아니라 AICE Associate 시험 유형을 참고해 직접 만든 **개인 학습용 예상문제**입니다. 데이터는 모두 가상으로 생성했어요.
시험 정보는 [AICE 공식 홈페이지](https://aice.study)에서 확인하세요.

도움이 됐다면 ⭐ 하나 눌러 주세요!
