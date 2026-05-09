# Problem 1. Marabou Resources Directory 탐색 노트

조사일: 2026-05-09

대상 저장소: https://github.com/NeuralNetworkVerification/Marabou

대상 디렉터리: https://github.com/NeuralNetworkVerification/Marabou/tree/master/resources

## 조사 목적

이번 조사의 목적은 Marabou 저장소의 `resources` 디렉터리에 어떤 예제 모델, 데이터셋, 입력 명세, 검증 property가 들어 있는지 파악하는 것이다. 과제의 Problem 2에서는 이 디렉터리에 이미 포함되지 않은 외부 모델을 선택해야 하므로, 먼저 Marabou가 어떤 형식과 규모의 모델을 예제로 제공하는지 확인할 필요가 있다.

## 전체 구조 요약

`resources/README.md`에 따르면 이 폴더는 Marabou tool paper의 실험 결과를 재현하기 위한 파일들을 포함한다. 특히 `nnet` 폴더에는 ACAS XU, CollisionAvoidance, TwinStreams benchmark가 들어 있고, `properties` 폴더에는 ACAS XU용 property 파일과 CollisionAvoidance/TwinStreams용 `builtin_property.txt`가 들어 있다.

GitHub tree를 기준으로 확인한 주요 top-level 항목은 다음과 같다.

| 경로 | 역할 |
| --- | --- |
| `resources/nnet` | `.nnet` 형식 신경망 benchmark |
| `resources/onnx` | `.onnx` 형식 모델과 ONNX layer/operator 테스트 모델 |
| `resources/properties` | `.txt` 형식의 입력/출력 제약 property |
| `resources/onnx/vnnlib` | VNNLIB 형식 property와 작은 ONNX 테스트 모델 |
| `resources/keras` | Keras `.h5` 모델 |
| `resources/bnn_queries` | Binary Neural Network 관련 query 자료 |
| `resources/mps` | LP feasibility/infeasibility 확인용 `.mps` 예제 |
| `resources/target` | 더 어려운 target query 예제 |
| `resources/SplitAndConquerGuide.ipynb` | Split-and-Conquer 실행 방식 안내 notebook |
| `resources/runMarabou.py` | 예제 실행을 돕는 Python script |

조사 시점의 GitHub tree 기준으로 확인한 주요 확장자 수는 다음과 같다.

| 확장자 | 파일 수 | 의미 |
| --- | ---: | --- |
| `.nnet` | 634 | Marabou/Reluplex 계열 benchmark 모델 |
| `.onnx` | 119 | ONNX 모델과 operator 테스트 모델 |
| `.txt` | 87 | 입력/출력 제약 property |
| `.vnnlib` | 15 | VNNLIB 형식 property |
| `.h5` | 4 | Keras 모델 |
| `.ipq` | 2 | query 파일 |
| `.mps` | 2 | LP 문제 예제 |

확장자별로 보면 `.nnet` 파일이 가장 많고, 그 다음으로 `.onnx`, `.txt`, `.vnnlib` property 파일이 많이 보인다. 이는 Marabou가 모델 파일과 property 파일을 분리해서 다루는 방식에 익숙해지는 데 좋은 예시가 된다.

## 제공되는 모델 유형

### `.nnet` 모델

`resources/nnet`은 Marabou의 전통적인 benchmark 형식인 `.nnet` 파일들을 포함한다. 주요 하위 폴더는 다음과 같다.

| 폴더 | 내용 |
| --- | --- |
| `acasxu` | ACAS XU aircraft collision avoidance benchmark |
| `coav` | CollisionAvoidance benchmark |
| `mnist` | MNIST 분류용 fully connected network |
| `twin` | TwinStreams benchmark |
| `fc_2-2-3.nnet` | 매우 작은 fully connected 예제 모델 |

`acasxu`에는 `ACASXU_experimental_v2a_1_1.nnet`처럼 이름 붙은 여러 네트워크가 들어 있다. ACAS XU는 항공기 충돌 회피 시스템을 위한 대표적인 신경망 검증 benchmark로, 입력 범위와 출력 간 비교 조건을 통해 안전성 성질을 확인한다.

`mnist`에는 `mnist10x10.nnet`, `mnist10x20.nnet`, `mnist20x20.nnet`, `mnist2x256.nnet` 등 작은 fully connected MNIST 네트워크들이 있다. Problem 2에서 너무 큰 CNN을 피하고 작은 모델을 선택해야 한다는 점을 생각하면, 이 폴더는 적절한 모델 규모를 판단하는 기준으로 쓸 수 있다.

### `.onnx` 모델

`resources/onnx`는 Marabou가 현재 중심적으로 지원하는 ONNX 모델 예제를 포함한다. Marabou README는 ONNX를 main network format으로 설명하고, 추가로 `.nnet` 및 TensorFlow 형식도 지원한다고 설명한다.

주요 예시는 다음과 같다.

| 경로 | 내용 |
| --- | --- |
| `onnx/acasxu` | ACAS XU 모델의 ONNX 버전 |
| `onnx/cifar10` | CIFAR-10 관련 ONNX 모델 |
| `onnx/layer-zoo` | ONNX layer/operator별 단일 노드 테스트 모델 |
| `onnx/concat`, `onnx/split`, `onnx/resize` | 특정 ONNX 연산 테스트 모델 |
| `mnist2x10.onnx`, `mnist2x5_sigmoid.onnx`, `mnist5x20_leaky_relu.onnx` | MNIST 계열 작은 모델 |
| `model-german-traffic-sign-fast.onnx`, `traffic-classifier64.onnx` | traffic sign 분류 모델 |
| `KJ_TinyTaxiNet.onnx` | TinyTaxiNet 예제 |
| `self-attention-mnist-pgd-medium-sim.onnx` | MNIST self-attention 계열 모델 |

`onnx/acasxu/README.md`에 따르면 이 ONNX ACAS XU 모델들은 VNN-COMP 2021 benchmark에서 가져온 뒤 `.nnet` 네트워크 이름과 맞추도록 rename된 것이다.

`onnx/cifar10/README.md`에는 CIFAR-10 네트워크가 특정 논문에서 평가된 모델이며, `simp` 버전은 ONNX simplifier로 shape, concat, unsqueeze 같은 연산을 제거한 버전이라고 되어 있다. 이 점은 Marabou에서 복잡한 ONNX 연산이 있으면 모델 단순화가 필요할 수 있음을 보여준다.

`onnx/layer-zoo/README.md`는 이 폴더가 현재 지원되는 layer를 하나씩 포함한 single node neural network 모음이라고 설명한다. `add`, `batchnorm`, `conv`, `dropout`, `flatten`, `gemm`, `leakyRelu`, `matmul`, `maxpool`, `relu`, `sigmoid`, `tanh` 같은 연산별 테스트 파일이 들어 있다.

### `.h5`, `.mps`, `.ipq` 등 기타 자료

`resources/keras`에는 `cnn_max_mnist2.h5`, `cnn_max_mnist3.h5`, `fc_2-2sigmoids-3.h5`, `robust_mnist_sigmoid_linear.h5` 같은 Keras 모델이 들어 있다. 이는 ONNX 또는 다른 형식으로 변환하기 전의 원본 모델 자료로 볼 수 있다.

`resources/mps`에는 `lp_feasible_1.mps`, `lp_infeasible_1.mps`가 있다. 이는 일반 신경망 모델이라기보다는 solver가 선형 제약 문제를 다루는 데 필요한 LP 형식 예제에 가깝다.

`resources/bnn_queries`와 `resources/target`에는 `.ipq` query 파일이 포함되어 있다. `target/README.md`는 이 폴더를 언젠가 효율적으로 풀 수 있을 challenging query 모음이라고 설명한다.

## 포함된 데이터셋과 입력 명세

`resources` 디렉터리에 있는 자료들은 완전한 학습 데이터셋 전체라기보다는, 검증 실험에 필요한 작은 benchmark 모델과 property 파일에 가깝다.

주요 입력 도메인은 다음과 같다.

| 도메인 | 위치 | 입력 명세의 성격 |
| --- | --- | --- |
| ACAS XU | `nnet/acasxu`, `onnx/acasxu`, `properties/acas_property_[1-4].txt` | 항공기 충돌 회피 상황의 연속값 입력 범위 |
| CollisionAvoidance | `nnet/coav`, `properties/builtin_property.txt` | 충돌 회피 benchmark, property 일부가 final layer에 내장됨 |
| TwinStreams | `nnet/twin`, `properties/builtin_property.txt` | TwinStreams benchmark, margin과 layer/width 설정이 파일명에 반영됨 |
| MNIST | `nnet/mnist`, `onnx`의 MNIST 모델, `properties/mnist` | 이미지 픽셀별 perturbation bound와 targeted attack property |
| CIFAR-10 | `onnx/cifar10` | CIFAR-10 분류 모델, 일부 모델은 ONNX simplifier 적용 |
| traffic sign | `onnx/model-german-traffic-sign-fast.onnx`, `onnx/traffic-classifier64.onnx` | German traffic sign 계열 분류 모델 |
| TinyTaxiNet | `onnx/KJ_TinyTaxiNet.onnx` | 작은 taxi navigation 관련 네트워크 예제 |

MNIST property 파일 이름은 `image3_target6_epsilon0.05.txt`처럼 구성되어 있다. 이는 특정 이미지 주변의 epsilon 범위를 입력 제약으로 두고, 특정 target class와 출력 점수를 비교하는 targeted robustness query임을 나타낸다.

## 제공되는 검증 query와 property

Marabou의 일반적인 query는 네트워크 파일과 property 파일의 조합으로 구성된다. Marabou README의 예시는 다음과 같다.

```bash
Marabou resources/nnet/acasxu/ACASXU_experimental_v2a_2_7.nnet resources/properties/acas_property_3.txt
```

이 예시는 ACAS XU `.nnet` 모델 하나와 `acas_property_3.txt`를 함께 입력해서 검증을 실행한다.

대표 property 유형은 다음과 같다.

| property 위치 | 내용 |
| --- | --- |
| `properties/acas_property_1.txt`부터 `acas_property_4.txt` | ACAS XU 입력 범위와 출력 간 비교 제약 |
| `properties/builtin_property.txt` | `y0 <= 0` 형태의 간단한 출력 제약, CollisionAvoidance와 TwinStreams에서 사용 |
| `properties/mnist/image*_target*_epsilon*.txt` | MNIST 이미지 주변 epsilon perturbation과 target class 비교 |
| `onnx/vnnlib/*.vnnlib` | VNNLIB 형식으로 작성된 표준화된 property 예제 |
| `target/*.ipq`, `fashion.ipq` | 더 어려운 query 또는 특정 실험용 query |

예를 들어 `acas_property_3.txt`는 입력 변수 `x0`부터 `x4`까지의 범위를 제한하고, 출력 변수에 대해 `y0`이 다른 출력보다 작거나 같다는 식의 선형 비교 조건을 둔다. MNIST property 파일들은 784개 픽셀 입력 각각에 대해 하한과 상한을 설정하고, 마지막에 target class 출력과 다른 class 출력의 비교 조건을 둔다.

## Problem 2와 연결되는 관찰

이 탐색에서 가장 중요한 점은 Marabou 예제가 대부분 작은 `.nnet` 또는 `.onnx` 모델을 중심으로 구성되어 있다는 것이다. 특히 ACAS XU, MNIST fully connected network, 작은 ONNX layer test 모델이 많이 들어 있다.

따라서 Problem 2에서는 다음 기준을 따르는 것이 좋다.

1. `resources`에 이미 있는 ACAS XU, MNIST 예제 모델을 그대로 쓰지 않는다.
2. Marabou가 잘 처리할 수 있도록 작은 fully connected model 또는 단순한 CNN을 고른다.
3. 가능하면 ONNX로 export해서 `maraboupy`에서 읽는다.
4. 모델이 복잡한 ONNX 연산을 포함하면 `onnx-simplifier` 같은 도구로 단순화가 필요한지 확인한다.
5. 검증 query는 처음부터 큰 epsilon을 쓰지 않고, 작은 epsilon의 local robustness query로 시작한다.

현재 자료만 보면 외부 모델 후보로는 작은 tabular classification 모델이나 scikit-learn/PyTorch로 직접 학습한 작은 MLP를 ONNX로 변환하는 방식이 적합해 보인다. 이렇게 하면 Marabou resources에 포함된 모델과 겹치지 않으면서도, property 구성과 실행 시간이 과제 범위 안에 들어올 가능성이 높다.

## 참고한 자료

- Marabou resources directory: https://github.com/NeuralNetworkVerification/Marabou/tree/master/resources
- Marabou resources README: https://github.com/NeuralNetworkVerification/Marabou/blob/master/resources/README.md
- Marabou main README: https://github.com/NeuralNetworkVerification/Marabou/blob/master/README.md
- ONNX ACAS XU README: https://github.com/NeuralNetworkVerification/Marabou/blob/master/resources/onnx/acasxu/README.md
- ONNX CIFAR-10 README: https://github.com/NeuralNetworkVerification/Marabou/blob/master/resources/onnx/cifar10/README.md
- ONNX layer-zoo README: https://github.com/NeuralNetworkVerification/Marabou/blob/master/resources/onnx/layer-zoo/README.md
- target README: https://github.com/NeuralNetworkVerification/Marabou/blob/master/resources/target/README.md
