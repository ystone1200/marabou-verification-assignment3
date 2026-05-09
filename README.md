# Assignment 3: Marabou Wine MLP Verification

이 저장소는 Marabou를 사용해 작은 외부 신경망 모델의 local robustness를 검증하기 위한 과제 작업물이다.

Problem 1 조사 내용은 `docs/resources_exploration.md`에 정리했다. Problem 2에서는 `sklearn.datasets.load_wine()` 데이터셋으로 학습한 작은 MLP를 ONNX로 export하고, Marabou로 출력 class가 바뀌는 반례가 존재하는지 확인한다.

## 환경

Marabou README에 따르면 `maraboupy`는 Python 3.8부터 3.11까지 지원한다. 이 프로젝트는 Python 3.10 또는 3.11 환경을 권장한다.

Windows PC에서는 WSL2 Ubuntu 환경에서 실행하는 것을 권장한다. Marabou는 현재 Windows native build를 공식 지원하지 않는다고 안내하고 있다.

## 설치

Ubuntu 24.04에서 기본 Python이 3.12인 경우, Python 3.11 환경을 따로 만든 뒤 실행한다. 이 작업에서는 다음 micromamba 환경을 사용했다.

```bash
~/.local/bin/micromamba create -y -p ~/.local/share/mamba-envs/assignment3 -c conda-forge python=3.11 pip
~/.local/bin/micromamba run -p ~/.local/share/mamba-envs/assignment3 python -m pip install -r requirements.txt
```

Python 3.10 또는 3.11이 이미 준비되어 있다면 일반 venv를 사용해도 된다.

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

`wine_mlp_metadata.json`에는 scaler 정보, 정확도, ONNX/PyTorch 출력 차이, reference test sample 정보가 기록된다.

## Marabou 검증 실행

모델을 생성한 뒤 다음 명령을 실행한다.

```bash
python test.py
```

기본 실행에서는 test set 전체를 다시 구성한 뒤, 올바르게 분류된 sample 중 output logit margin이 가장 작은 sample 3개를 선택한다. 그런 다음 normalized feature space에서 다음 epsilon 값을 확인한다.

```text
0.01, 0.05, 0.1, 0.3, 0.5, 1.0
```

특정 epsilon, sample 수, timeout을 지정하려면 다음처럼 실행한다.

```bash
python test.py --sample-count 5 --epsilons 0.01 0.05 0.1 0.2 0.3 --timeout 120
```

metadata에 저장된 reference sample 하나만 다시 검증하려면 다음처럼 실행한다.

```bash
python test.py --sample-source stored
```

결과는 화면에 출력되고, 동시에 다음 파일로 저장된다.

```text
results/wine_marabou_results.json
```

`SAT` 결과가 나오면 Marabou가 찾은 counterexample input과 output도 JSON에 함께 저장된다.

## 주요 검증 결과

최종 실행에서는 low-margin correct test sample 3개를 선택해 epsilon sweep을 수행했다.

| Sample | Logit margin | Largest tested robust epsilon | First non-robust epsilon |
| --- | ---: | ---: | ---: |
| `low_margin_1` | 0.002380 | 없음 | 0.01 |
| `low_margin_2` | 0.293360 | 없음 | 0.01 |
| `low_margin_3` | 4.312881 | 0.1 | 0.3 |

`low_margin_3`은 epsilon `0.01`, `0.05`, `0.1`에서 `UNSAT`으로 robust했고, epsilon `0.3`부터 `SAT` counterexample이 발견되었다. 이 결과는 `results/wine_marabou_results.json`에 저장되어 있다.

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
report.pdf                     최종 1-2페이지 보고서
models/                        생성된 ONNX 모델 저장 위치
results/                       Marabou 검증 결과 저장 위치
```
