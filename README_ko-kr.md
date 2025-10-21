[Go to English Version](./README.md)

# COMPOSABLE TRAINING TEMPLATE README

이 문서는 `composable-training-template` 프로젝트를 사용하여 새로운 딥러닝 실험을 구성하고 실행하는 방법을 안내합니다.

## 1. 프로젝트 개요

이 프로젝트는 [Hydra](https://hydra.cc/)와 [PyTorch Lightning](https://www.pytorchlightning.ai/)을 기반으로 한 유연하고 재사용 가능한 딥러닝 실험 프레임워크입니다. 핵심 목표는 다음과 같습니다.

  * **설정 기반 실험**: 모델 아키텍처, 데이터셋, 손실 함수 등 실험의 모든 요소를 YAML 설정 파일로 관리합니다.
  * **모듈화된 구조**: 모델의 각 부분을 독립적인 컴포넌트로, 학습 로직(손실 함수 등)을 '전략(Strategy)'으로 분리하여 코드 재사용성을 극대화합니다.
  * **재현성 및 확장성**: 설정 파일을 통해 실험 조건을 명확하게 정의하므로, 누구나 쉽게 실험을 재현하고 새로운 모델이나 방법론으로 확장할 수 있습니다.

## 2. 핵심 개념

프로젝트를 이해하기 위해 알아야 할 네 가지 핵심 요소가 있습니다.

#### 2.1. Experiment (`configs/experiment`)

실험의 모든 구성요소를 조합하는 최상위 설정 파일입니다. 하나의 실험은 **Model**, **Datamodule**, **Strategy**, **Trainer**, **Optimizer** 등의 설정을 참조하여 완성됩니다.

**예시: `configs/experiment/sentiment-analysis/misa2020/base.yaml`**

```yaml
defaults:
  - /trainer: supervised/regression
  - /datamodule: cmu-mosi/msa
  - /model: sentiment-analysis/misa2020/base
  - /strategy: sentiment-analysis/misa2020/base
  - /optimizer: adamw_default
  - _self_

# ... (실험별 하이퍼파라미터)
````

#### 2.2. GraphModel (`source/models/graph_model.py`)

이 프로젝트의 핵심 모델링 방식은 복잡한 모델 아키텍처를 **비순환 그래프(DAG)** 형태로 정의합니다. **구조적 순수성을 위해, 모든 Operation(Op, 예: `Add`, `Concat`)은 반드시 `module`을 가진 독립적인 노드로 분리**되어야 합니다.

노드의 입력은 **단일 실행**과 **반복 실행**의 두 가지 필드로 정의되며, 두 필드 모두 유연한 인자 전달(`*args`, `**kwargs`)을 지원합니다.

| 필드 | 역할 | YAML 값 형태 | 파이썬 인자 전달 |
| :--- | :--- | :--- | :--- |
| **`inputs`** | **단일 실행**: 모듈을 한 번 실행합니다. (Op 노드의 출력, 또는 다음 레이어의 입력) | String, List, 또는 Dict | `module(*args)` 또는 `module(**kwargs)` |
| **`inputs_for`** | **반복 실행**: 모듈을 입력 항목별로 반복 실행하고, 결과를 딕셔너리로 반환합니다. | Dict (내부에 String, List, Dict 구조) | `module(*args)` 또는 `module(**kwargs)` (반복 적용) |

**예시: Op 노드 분리 및 `foreach` 활용**

이 예시는 `Concat` 연산을 `_op_fusion` 노드로 분리하고, `shared_encoders` 노드에서 `inputs_for`를 사용해 동일한 Linear 모듈을 각 모달리티에 적용하는 방법을 보여줍니다.

```yaml
graph_cfg:
  # 1. Op 노드 분리: Concat Operation을 독립적인 노드로 정의
  _op_fusion:
    module: { class_path: models.components.ops.Concat }
    # List를 사용하여 *args 인자를 전달: Concat(encoded_t, encoded_v) 호출
    inputs: ['encoded_t', 'encoded_v'] 
  
  # 2. 주 모듈 노드: Op 노드의 결과를 입력으로 받음 (단일 실행)
  final_mlp:
    module: { class_path: models.components.feed_forward.MLP, init_args: { ... } }
    # 이전 Op 노드의 출력을 단일 String으로 받음
    inputs: '_op_fusion' 
    
  # 3. 반복 실행 예시 (foreach)
  shared_encoders:
    module: { class_path: models.components.feed_forward.Linear, init_args: { ... } }
    inputs_for:
      # Key(text)와 Value(Dict)를 사용하여 Linear 모듈을 **kwargs로 호출
      text:
        x: 'projected_t'
        mask: 't_mask' 
      # Key(vision)와 Value(List)를 사용하여 모듈을 *args로 호출
      vision:
        - 'projected_v'
        - 'v_mask'
```

#### 2.3. Strategy (`source/strategies/`)

손실 함수 계산, 특정 학습 기법(예: BYOL) 등 모델의 학습 방식과 관련된 로직을 담당하는 모듈입니다. `ExperimentModule`은 `configs/strategy/**/*.yaml`에 정의된 여러 Strategy를 조합하여 최종 손실을 계산합니다.

  * 모든 Strategy는 `BaseStrategy` 클래스를 상속받아 `calculate` 메소드를 구현해야 합니다.
  * 이를 통해 손실 함수를 모델 아키텍처와 분리하여 독립적으로 개발하고 재사용할 수 있습니다.

**예시: MISA의 `difference_loss` Strategy 설정**

```yaml
difference_loss:
  module:
    class_path: strategies.custom.misa2020.DifferenceStrategy 
    init_args:
      weight: 0.3 # 손실 가중치
  inputs: # Strategy의 calculate 메소드에 전달될 인자
    shared_t: model_outputs.shared.t
    private_t: model_outputs.private_t
    # ...
```

#### 2.4. ExperimentModule (`source/pl_modules/experiment_module.py`)

PyTorch Lightning의 `LightningModule`을 상속받은 클래스로, Model과 Strategy를 결합하여 실제 학습(`training_step`), 검증(`validation_step`), 테스트(`test_step`)를 수행합니다. 이 모듈은 설정된 Strategy들을 순회하며 각 손실을 계산하고 합산하여 최종 손실을 구합니다.

## 3\. 실험 실행 방법

### 3.1. 기본 실험 실행

`source/train_supervised.py`를 실행하여 학습을 시작합니다. Hydra는 `configs/train_supervised.yaml` 파일을 기본 진입점으로 사용합니다.

```bash
python source/train_supervised.py
```

### 3.2. 다른 실험 실행

`experiment` 인자를 변경하여 다른 실험을 실행할 수 있습니다. 예를 들어, `misa2020/base` 실험을 실행하려면 다음과 같이 입력합니다.

```bash
python source/train_supervised.py experiment=sentiment-analysis/misa2020/base
```

## 4\. 새로운 실험 추가하기

새로운 데이터셋과 모델로 실험을 추가하는 과정은 다음과 같습니다.

### **1단계: 데이터셋 준비 및 설정**

1.  **데이터셋 클래스 준비**: PyTorch의 `Dataset` 클래스를 구현합니다. 이미 구현된 `LazyPklDataset`이나 `LazyTorchDataset`을 사용할 수 있다면 이 단계를 건너뛸 수 있습니다.

2.  **Collator 구현 (선택적)**: 데이터 배치를 특별한 방식으로 구성해야 한다면, `source/utils/collator.py`에 새로운 Collator 클래스를 작성합니다.

3.  **Datamodule 설정 파일 생성**: `configs/datamodule/` 아래에 새로운 YAML 파일을 생성합니다. 이 파일은 `UniversalDataModule`을 사용하며, 데이터셋 경로, 배치 크기 등을 정의합니다.

    **예시: `configs/datamodule/my_dataset.yaml`**

    ```yaml
    class_path: datamodules.UniversalDataModule
    init_args:
      train_dataset_config:
        class_path: dataset.LazyPklDataset
        init_args: { data_path: /path/to/my_train_data.pkl }

      test_dataset_config:
        # ...

      collate_fn_config:
        _target_: utils.collator.MyCustomCollator # 필요시

      batch_size: 32
    ```

### **2단계: 모델 아키텍처 정의**

1.  **모델 컴포넌트 추가 (필요시)**: 모델에 필요한 새로운 레이어나 연산이 있다면 `source/models/components/` 디렉토리 아래에 Python 파일로 구현합니다.

2.  **모델 설정 파일 생성**: `configs/model/` 아래에 새로운 YAML 파일을 생성합니다. `GraphModel`을 사용하여 모델의 노드와 연결 관계를 정의합니다.

    **예시: `configs/model/my_model.yaml` (업데이트된 구조)**

    ```yaml
    class_path: models.GraphModel
    init_args:
      inputs: ['input_a', 'input_b']
      outputs: ['logits']

      graph_cfg:
        encoded_a:
          module: { class_path: models.components.feed_forward.Linear, init_args: { ... } }
          inputs: 'input_a' # 단일 String 입력
        
        # Op 노드: Concat Operation을 독립적으로 정의
        _op_concat:
          module: { class_path: models.components.ops.Concat }
          inputs: ['encoded_a', 'encoded_b'] # List 입력 (*args)

        # 주 모듈 노드: Op 노드의 출력을 받음
        fused_representation:
          module: { class_path: models.components.feed_forward.Linear, init_args: { ... } }
          inputs: '_op_concat' # Op 노드 출력(단일 String)을 입력으로 사용

        logits:
          module: { class_path: torch.nn.Sigmoid }
          inputs: 'fused_representation'
    ```

### **3단계: 학습 전략(손실 함수) 정의**

1.  **Strategy 클래스 구현 (필요시)**: 새로운 손실 함수나 학습 로직이 필요하다면 `source/strategies/` 아래(주로 `custom/`)에 `BaseStrategy`를 상속받는 클래스를 구현합니다.

2.  **Strategy 설정 파일 생성**: `configs/strategy/` 아래에 새로운 YAML 파일을 생성합니다. 이 파일은 학습에 사용할 하나 이상의 Strategy를 정의하고, 각 Strategy에 어떤 모델 출력을 입력으로 줄지 명시합니다.

    **예시: `configs/strategy/my_task.yaml`**

    ```yaml
    classification_loss:
      module:
        class_path: strategies.supervised.BinaryClassificationStrategy
        init_args:
          weight: 1.0
      inputs:
        logits: model_outputs.logits
        labels: batch.labels

    my_custom_loss:
      module:
        class_path: strategies.custom.MyAwesomeLossStrategy
        init_args:
          lambda_weight: 0.5
      inputs:
        representation: model_outputs.fused_representation
    ```

### **4단계: 실험 종합 설정**

`configs/experiment/` 아래에 위에서 만든 설정들을 종합하는 최종 실험 파일을 생성합니다.

**예시: `configs/experiment/my_experiment.yaml`**

```yaml
defaults:
  - /trainer: default # 또는 supervised/regression 등
  - /datamodule: my_dataset
  - /model: my_model
  - /strategy: my_task
  - /optimizer: adamw_default
  - _self_

trainer:
  max_epochs: 100

optimizer:
  lr: 5e-5
```

### **5단계: 새로운 실험 실행**

이제 커맨드 라인에서 새로 만든 실험을 이름으로 호출하여 실행할 수 있습니다.

```bash
python source/train_supervised.py experiment=my_experiment
```

