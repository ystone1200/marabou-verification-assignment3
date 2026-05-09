# Problem 2. 외부 모델과 데이터셋 선택

## 선택한 방향

Problem 2에서는 `Wine dataset`으로 학습한 작은 fully connected neural network를 사용한다. 모델은 PyTorch로 학습하고 ONNX 형식으로 export한다.

예정 구조는 다음과 같다.

| 항목 | 선택 |
| --- | --- |
| 데이터셋 | `sklearn.datasets.load_wine()` |
| 입력 차원 | 13개 feature |
| 출력 class | 3개 wine cultivar class |
| 모델 형식 | ONNX |
| 모델 구조 | `13 -> 16 -> ReLU -> 8 -> ReLU -> 3` |
| 검증 property | 특정 test sample 주변의 local robustness |

## 선택 이유

Marabou 과제는 `resources` 디렉터리에 이미 포함되어 있지 않은 외부 모델을 요구한다. `resources`에는 ACAS XU, MNIST, CIFAR-10, traffic sign, TinyTaxiNet 등의 예제가 있지만 Wine dataset 기반 모델은 포함되어 있지 않다.

Wine dataset은 Iris보다 feature가 많아서 너무 단순하지 않으면서도, MNIST나 CIFAR-10처럼 입력 차원이 크지 않다. 입력 feature가 13개라 Marabou의 SMT 기반 검증이 처리하기에 비교적 적절한 규모라고 판단했다.

## pretrained 모델 대신 직접 학습하는 이유

이미 학습된 Wine 관련 모델을 찾아보면, 공개된 모델 중 상당수는 RandomForestClassifier 같은 tree 기반 모델이거나, Marabou에서 다루기 어려운 큰 모델이다. Marabou는 ONNX를 주요 형식으로 지원하지만, 과제의 목적상 neural network verification을 보여주는 것이 중요하므로 작은 MLP를 직접 학습하는 쪽이 더 적합하다.

직접 학습하면 모델 구조, activation, 입력 전처리, ONNX export 방식을 통제할 수 있다. 특히 마지막 layer에 softmax를 넣지 않고 logits를 출력하게 만들면, Marabou에서 출력 class 간 선형 비교 constraint를 구성하기 쉽다.

## 검증 query 계획

검증 대상은 test set에서 올바르게 분류된 sample 하나로 정한다. 이 sample의 normalized input을 `x`라고 할 때, 각 feature에 대해 다음 입력 범위를 둔다.

```text
x_i - epsilon <= x'_i <= x_i + epsilon
```

모델의 원래 예측 class를 `c`라고 하면, 다른 class `j`에 대해 다음 반례 조건을 각각 확인한다.

```text
y_j >= y_c
```

Marabou에서 이 조건이 `UNSAT`이면 해당 epsilon 범위 안에서 class가 바뀌는 반례가 없다는 뜻이다. `SAT`이면 Marabou가 찾은 입력이 원래 class보다 다른 class를 더 크게 만드는 반례라는 뜻이다.

## 예상되는 장점과 위험

장점은 입력 차원이 작고 모델 구조가 단순해서 검증 시간이 과도하게 길어질 가능성이 낮다는 점이다. 또한 ONNX 모델의 연산이 `Gemm`과 `Relu` 중심이므로 Marabou 호환성 측면에서도 안정적이다.

위험은 normalized feature space에서 epsilon을 해석해야 한다는 점이다. 따라서 보고서에서는 검증이 원본 feature가 아니라 `StandardScaler`가 적용된 입력 공간에서 수행되었다고 명확히 설명해야 한다.
