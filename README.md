[한국어 버전으로 가기](./README_ko-kr.md)

# COMPOSABLE TRAINING TEMPLATE README

This document provides a guide on how to configure and run new deep learning experiments using the `composable-training-template` project.

## 1. Project Overview

This project is a flexible and reusable deep learning experimental framework based on [Hydra](https://hydra.cc/) and [PyTorch Lightning](https://www.pytorchlightning.ai/). The core objectives are as follows:

  * **Configuration-Based Experiments**: All components of an experiment, such as model architecture, datasets, and loss functions, are managed through YAML configuration files.
  * **Modular Structure**: It maximizes code reusability by separating each part of the model into independent components and the training logic (like loss functions) into "Strategies".
  * **Reproducibility and Extensibility**: Since experimental conditions are clearly defined through configuration files, anyone can easily reproduce experiments and extend them with new models or methodologies.

## 2. Core Concepts

To understand the project, you need to be familiar with four key elements.

#### 2.1. Experiment (`configs/experiment`)

This is the top-level configuration file that combines all the components of an experiment. A single experiment is completed by referencing configurations for **Model**, **Datamodule**, **Strategy**, **Trainer**, and **Optimizer**.

**Example: `configs/experiment/sentiment-analysis/misa2020/base.yaml`**

```yaml
defaults:
  - /trainer: supervised/regression
  - /datamodule: cmu-mosi/msa
  - /model: sentiment-analysis/misa2020/base
  - /strategy: sentiment-analysis/misa2020/base
  - /optimizer: adamw_default
  - _self_

# ... (Experiment-specific hyperparameters)
````

#### 2.2. GraphModel (`source/models/graph_model.py`)

This is the core modeling approach of the project, defining complex model architectures in the form of a **Directed Acyclic Graph (DAG)**. To ensure **structural purity**, all primitive **Operations (Ops)** (e.g., `Add`, `Concat`) must be defined as **independent nodes** with their own `module`.

Node inputs are defined by two mutually exclusive fields for **single execution** and **repeated execution**, both supporting flexible argument passing (`*args`, `**kwargs`).

| Field | Role | YAML Value Type | Python Calling Convention |
| :--- | :--- | :--- | :--- |
| **`inputs`** | **Single Execution**: Executes the module once. (Used for layer inputs or receiving Op results) | String, List, or Dict | `module(*args)` or `module(**kwargs)` |
| **`inputs_for`** | **Repeated Execution**: Executes the module repeatedly for each item (e.g., each modality) and returns a dictionary of results. | Dict (Keyed by output suffix, valued by String, List, or Dict input structures) | `module(*args)` or `module(**kwargs)` (repeatedly) |

**Example: Op Node Separation and `foreach` Usage**

This example demonstrates separating the `Concat` operation into an `_op_fusion` node and using `inputs_for` to apply the same Linear module to multiple modalities.

```yaml
graph_cfg:
  # 1. Op Node Separation: Define Concat Operation as an independent node
  _op_fusion:
    module: { class_path: models.components.ops.Concat }
    # List used for *args input: Concat(encoded_t, encoded_v) call
    inputs: ['encoded_t', 'encoded_v'] 
  
  # 2. Main Module Node: Receives the Op Node output (Single Execution via 'inputs')
  final_mlp:
    module: { class_path: models.components.feed_forward.MLP, init_args: { ... } }
    # Receives the output of the previous Op node as a single string input
    inputs: '_op_fusion' 
    
  # 3. Repeated Execution Example (foreach)
  shared_encoders:
    module: { class_path: models.components.feed_forward.Linear, init_args: { ... } }
    inputs_for:
      # Key(text) and Value(Dict) are used to call the Linear module with **kwargs
      text:
        x: 'projected_t'
        mask: 't_mask' 
      # Key(vision) and Value(List) are used to call the module with *args
      vision:
        - 'projected_v'
        - 'v_mask'
```

#### 2.3. Strategy (`source/strategies/`)

This module is responsible for logic related to the model's training method, such as loss function calculation and specific training techniques (e.g., BYOL). The `ExperimentModule` combines multiple Strategies defined in `configs/strategy/**/*.yaml` to calculate the final loss.

  * All Strategies must inherit from the `BaseStrategy` class and implement the `calculate` method.
  * This allows the loss functions to be developed and reused independently from the model architecture.

**Example: Configuration for the `difference_loss` Strategy in MISA**

```yaml
difference_loss:
  module:
    class_path: strategies.custom.misa2020.DifferenceStrategy 
    init_args:
      weight: 0.3 # Loss weight
  inputs: # Arguments passed to the Strategy's calculate method
    shared_t: model_outputs.shared.t
    private_t: model_outputs.private_t
    # ...
```

#### 2.4. ExperimentModule (`source/pl_modules/experiment_module.py`)

This class, which inherits from PyTorch Lightning's `LightningModule`, combines the Model and Strategies to perform the actual training (`training_step`), validation (`validation_step`), and testing (`test_step`). This module iterates through the configured Strategies, calculates each loss, and sums them up to get the final loss.

## 3\. How to Run an Experiment

### 3.1. Running the Default Experiment

Execute `source/train_supervised.py` to start training. Hydra uses the `configs/train_supervised.yaml` file as the default entry point.

```bash
python source/train_supervised.py
```

### 3.2. Running a Different Experiment

You can run a different experiment by changing the `experiment` argument. For example, to run the `misa2020/base` experiment, enter the following:

```bash
python source/train_supervised.py experiment=sentiment-analysis/misa2020/base
```

## 4\. How to Add a New Experiment

The process for adding an experiment with a new dataset and model is as follows.

### **Step 1: Prepare and Configure the Dataset**

1.  **Prepare the Dataset Class**: Implement a `Dataset` class from PyTorch. You can skip this step if you can use the already implemented `LazyPklDataset` or `LazyTorchDataset`.

2.  **Implement a Collator (Optional)**: If you need a special way to construct data batches, write a new Collator class in `source/utils/collator.py`.

3.  **Create a Datamodule Config File**: Create a new YAML file under `configs/datamodule/`. This file will use `UniversalDataModule` and define the dataset paths, batch size, etc.

    **Example: `configs/datamodule/my_dataset.yaml`**

    ```yaml
    class_path: datamodules.UniversalDataModule
    init_args:
      train_dataset_config:
        class_path: dataset.LazyPklDataset
        init_args: { data_path: /path/to/my_train_data.pkl }

      test_dataset_config:
        # ...

      collate_fn_config:
        _target_: utils.collator.MyCustomCollator # If needed

      batch_size: 32
    ```

### **Step 2: Define the Model Architecture**

1.  **Add Model Components (If necessary)**: If your model requires new layers or operations, implement them as Python files under the `source/models/components/` directory.

2.  **Create a Model Config File**: Create a new YAML file under `configs/model/`. Use `GraphModel` to define the model's nodes and their connections.

    **Example: `configs/model/my_model.yaml` (Updated for new structure)**

    ```yaml
    class_path: models.GraphModel
    init_args:
      inputs: ['input_a', 'input_b']
      outputs: ['logits']

      graph_cfg:
        encoded_a:
          module: { class_path: models.components.feed_forward.Linear, init_args: { ... } }
          inputs: 'input_a' # Single String input
        
        # Op Node: Concat Operation defined as an independent node
        _op_concat:
          module: { class_path: models.components.ops.Concat }
          inputs: ['encoded_a', 'encoded_b'] # List input (*args)

        # Final Layer (receiving Op output)
        fused_representation:
          module: { class_path: models.components.feed_forward.Linear, init_args: { ... } }
          inputs: '_op_concat' # Op node output as single String input

        logits:
          module: { class_path: torch.nn.Sigmoid }
          inputs: 'fused_representation'
    ```

### **Step 3: Define the Training Strategy (Loss Functions)**

1.  **Implement a Strategy Class (If necessary)**: If a new loss function or training logic is needed, implement a class that inherits from `BaseStrategy` under `source/strategies/` (usually in `custom/`).

2.  **Create a Strategy Config File**: Create a new YAML file under `configs/strategy/`. This file defines one or more Strategies to be used for training and specifies which model outputs to pass as input to each Strategy.

    **Example: `configs/strategy/my_task.yaml`**

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

### **Step 4: Consolidate the Experiment Configuration**

Create a final experiment file under `configs/experiment/` that brings together all the configurations created above.

**Example: `configs/experiment/my_experiment.yaml`**

```yaml
defaults:
  - /trainer: default # or supervised/regression, etc.
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

### **Step 5: Run the New Experiment**

Now you can run your newly created experiment by calling it by name from the command line.

```bash
python source/train_supervised.py experiment=my_experiment
```
