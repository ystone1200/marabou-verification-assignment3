# Assignment 3: Marabou Wine MLP Verification

이 저장소는 Marabou를 사용해 작은 외부 신경망 모델의 local robustness를 검증하기 위한 과제 작업물이다.

Problem 1 조사 내용은 `docs/resources_exploration.md`에 정리했다. Problem 2에서는 `sklearn.datasets.load_wine()` 데이터셋으로 학습한 작은 MLP를 ONNX로 export하고, Marabou로 출력 class가 바뀌는 반례가 존재하는지 확인한다.

## 환경

Marabou README에 따르면 `maraboupy`는 Python 3.8부터 3.11까지 지원한다. 이 프로젝트는 Python 3.10 또는 3.11 환경을 권장한다.

Windows PC에서는 WSL2 Ubuntu 환경에서 실행하는 것을 권장한다. Marabou는 현재 Windows native build를 공식 지원하지 않는다고 안내하고 있다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 모델 학습 및 ONNX export

다음 명령은 Wine dataset을 불러와 작은 MLP를 학습하고, ONNX 모델과 metadata를 생성한다.

```bash
python src/train_wine_mlp.py
```

생성되는 파일은 다음과 같다.

```text
models/wine_mlp.onnx
models/wine_mlp_metadata.json
```

`wine_mlp_metadata.json`에는 scaler 정보, 정확도, 검증에 사용할 test sample, 원래 예측 class가 기록된다.

## Marabou 검증 실행

모델을 생성한 뒤 다음 명령을 실행한다.

```bash
python test.py
```

기본적으로 normalized feature space에서 다음 epsilon 값을 확인한다.

```text
0.01, 0.03, 0.05
```

특정 epsilon이나 timeout을 지정하려면 다음처럼 실행한다.

```bash
python test.py --epsilons 0.01 0.02 0.03 --timeout 120
```

결과는 화면에 출력되고, 동시에 다음 파일로 저장된다.

```text
results/wine_marabou_results.json
```

## 검증 property

선택된 test sample의 normalized input을 `x`라고 할 때, Marabou에는 각 feature에 대해 다음 입력 bound를 부여한다.

```text
x_i - epsilon <= x'_i <= x_i + epsilon
```

원래 예측 class를 `c`, 다른 class를 `j`라고 하면, 다음 조건을 만족하는 반례가 존재하는지 확인한다.

```text
y_j >= y_c
```

해석은 다음과 같다.

| Marabou 결과 | 의미 |
| --- | --- |
| `UNSAT` | 해당 epsilon 범위에서 class가 바뀌는 반례가 없음 |
| `SAT` | 해당 epsilon 범위 안에서 다른 class가 원래 class 이상이 되는 반례가 있음 |

## 파일 구조

```text
docs/resources_exploration.md  Marabou resources 탐색 노트
docs/model_selection.md        Wine MLP 선택 근거와 검증 계획
src/train_wine_mlp.py          Wine MLP 학습 및 ONNX export
test.py                        Marabou verification query 실행
requirements.txt               Python 의존성
models/                        생성된 ONNX 모델 저장 위치
```
