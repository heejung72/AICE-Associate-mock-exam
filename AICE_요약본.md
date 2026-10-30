# AICE Associate 최종 요약본

## 시험 흐름 (14문항)

```
import → 데이터 읽기 → 시각화 → 컬럼 삭제 → 결측치 → 이상치
→ 인코딩 → split → 스케일링 → 머신러닝 → 평가 → 딥러닝 → 학습 곡선
```

## 1. 기본 import

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

## 2. 데이터 읽기 / 확인

```python
df = pd.read_csv('파일.csv')       # json이면 pd.read_json()
df.head()
df.info()                           # 결측치, 자료형 확인
df.describe()
df['col'].value_counts()
df.isnull().sum()
```

## 3. 시각화

```python
sns.countplot(data=df, x='col')              # 범주별 개수
sns.histplot(data=df, x='col', bins=20, kde=True)   # 분포 + 곡선
sns.boxplot(data=df, x='범주', y='숫자')      # 이상치 확인
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')
```

## 4. 전처리

```python
# 컬럼 삭제
df1 = df.drop(columns=['id_col'])          # 또는 drop('col', axis=1)

# 결측치 개수
df.isnull().sum()          # 컬럼별
df.isnull().sum().sum()    # 전체

# 복사본 (원본 보존)
df_fill = df.copy()        # df_fill = df 는 같은 객체라 원본도 바뀜

# 결측치 채우기
df1['col'] = df1['col'].fillna(df1['col'].median())    # 숫자 → 중앙값
df1['col'] = df1['col'].fillna(df1['col'].mode()[0])   # 글자 → 최빈값 [0] 필수!
df1 = df1.dropna()                                      # 결측 행 전체 삭제

# 이상치 (IQR 정석)
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR
cond = (df['col'] < lower_fence) | (df['col'] > upper_fence)  # 울타리 밖 조건
df1 = df.drop(df[cond].index, axis=0)      # 이상치 행 삭제 → index 넘겨야 함
df1.reset_index(drop=True, inplace=True)   # 인덱스 초기화

# 값 바꾸기 (loc 사용 — 대입은 항상 loc!)
df1.loc[df1['col'].isin(['A', 'B']), 'col'] = 'other'

# 값 변환
df1['col'] = df1['col'].map({'A': 0, 'B': 1})
df1['col'] = df1['col'].astype(int)

# 중복 제거
df1 = df1.drop_duplicates()

# object 컬럼 목록
df1.select_dtypes(include='object').columns.tolist()

# 원핫 인코딩
df1 = pd.get_dummies(df1, columns=['col1', 'col2'], drop_first=True)
```

## 5. split

```python
from sklearn.model_selection import train_test_split

X = df1.drop('target', axis=1)
y = df1['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
# 회귀는 stratify 없이
```

> 반환 순서: **X_train, X_test, y_train, y_test**

## 6. 스케일링

```python
from sklearn.preprocessing import StandardScaler  # 또는 MinMaxScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)        # test는 transform만!
```

## 7. 머신러닝

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

회귀면 `DecisionTreeRegressor`, `RandomForestRegressor` 등으로 변경.

## 8. 평가

**분류**

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
```

**회귀**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print('MAE:', mean_absolute_error(y_test, pred))
print('R2:', r2_score(y_test, pred))
```

> 인자 순서: **(y_test, pred)**

## 9. 딥러닝

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(출력)                    # 아래 표 참고
])

model.compile(optimizer='adam', loss='손실함수', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=5)
mc = ModelCheckpoint('best.keras', monitor='val_loss', save_best_only=True)

history = model.fit(X_train, y_train,
                    epochs=50, batch_size=32,
                    validation_data=(X_test, y_test),
                    callbacks=[es, mc])
```

### 출력층 + loss 조합 (암기)

| 유형 | 출력층 | loss |
|---|---|---|
| 이진 분류 | `Dense(1, activation='sigmoid')` | `binary_crossentropy` |
| 다중 분류 | `Dense(클래스수, activation='softmax')` | `sparse_categorical_crossentropy` |
| 회귀 | `Dense(1)` | `mse` |

## 10. 학습 곡선

```python
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

## feature_importances_

```python
# 일반 모델에서
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf.feature_importances_

# GridSearchCV 사용 후
gs = GridSearchCV(rf, param_grid, cv=5)
gs.fit(X_train, y_train)
gs.best_estimator_.feature_importances_   # best_estimator_ 거쳐야 함
gs.best_params_                            # 최적 파라미터 확인
```

## GridSearchCV import

```python
from sklearn.model_selection import train_test_split, GridSearchCV
```

## 분석 함수 모음

```python
# 상관계수
df.corr(numeric_only=True)
corr.loc['col1', 'col2']   # 특정 두 컬럼 상관계수

# 그룹별 평균 (생존율 등)
df.groupby('Sex')['Survived'].mean()

# 교차표
pd.crosstab(df['Sex'], df['Survived'])
ct.loc['male', 1]   # 남성 생존자 수

# 정렬
df.sort_values('col', ascending=False).head(3)   # 내림차순

# 반올림
round(값, 소수자리수)
```

## 자주 틀리는 것

1. `train_test_split` 반환 순서 헷갈림
2. test에 `fit_transform` 사용 (transform만 써야 함)
3. 출력층 activation과 loss 짝 안 맞춤
4. 문제에서 지정한 변수명 안 지킴
5. `drop` 후 새 변수에 저장 안 함
6. `df.shape` — 괄호 없음 (메서드 아니고 속성)
7. 조건 필터링 시 각 조건마다 괄호 필수: `(df['A'] > 0) & (df['B'] == 1)`
8. `mode()` 뒤에 `[0]` 안 붙임
9. 이상치 삭제 시 `drop(df[cond].index, axis=0)` — `.index` 빠뜨림
10. 값 대입 시 `loc` 안 쓰면 원본 안 바뀜
11. `corr()`에 `numeric_only=True` 빠뜨림
