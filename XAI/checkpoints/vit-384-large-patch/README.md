---
library_name: transformers
license: apache-2.0
base_model: google/vit-large-patch32-384
tags:
- generated_from_trainer
metrics:
- precision
- recall
- f1
model-index:
- name: segmented-augmented
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# segmented-augmented

This model is a fine-tuned version of [google/vit-large-patch32-384](https://huggingface.co/google/vit-large-patch32-384) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6965
- Precision: 0.8085
- Recall: 0.8837
- F1: 0.8444

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | Precision | Recall | F1     |
|:-------------:|:-----:|:----:|:---------------:|:---------:|:------:|:------:|
| 0.0857        | 1.0   | 327  | 0.4359          | 0.8176    | 0.8040 | 0.8107 |
| 0.0164        | 2.0   | 654  | 0.5654          | 0.8043    | 0.8605 | 0.8315 |
| 0.0056        | 3.0   | 981  | 0.6437          | 0.8182    | 0.8671 | 0.8419 |
| 0.002         | 4.0   | 1308 | 0.6739          | 0.8055    | 0.8804 | 0.8413 |
| 0.003         | 5.0   | 1635 | 0.6965          | 0.8085    | 0.8837 | 0.8444 |


### Framework versions

- Transformers 4.44.2
- Pytorch 2.4.1+cu121
- Datasets 3.0.0
- Tokenizers 0.19.1
