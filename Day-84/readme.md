

# kFineTuning

## Introduction

This project demonstrates fine-tuning (transfer learning) on the CNN-VGG16 architecture using ImageNet weights for classification on the Oxford Flowers 17 dataset. Fine-tuning involves leveraging a pre-trained network's features learned from a larger dataset (like ImageNet) and applying them to a smaller dataset. It's especially effective in scenarios with limited data, such as medical imaging.

### Brief Summary

- Choose a pre-trained network (VGG-16 with ImageNet weights in this case).
- Trim the fully connected head of the CNN to obtain the CNN's body.
- Attach a new fully connected (FC) head to the CNN's body with the final output nodes matching the classes in the new dataset (17 classes in the Flowers17 dataset in this case).
- During the warm-up phase, freeze the body of the pre-trained CNN to allow back-propagation only till the new FC head, preserving the learned features.
- After achieving satisfactory results in the warm-up phase, unfreeze parts of the CNN's body to fine-tune the model further.


## Commands

```shell
$ python finetune.py --dataset flowers17/ --model myModel.model 
```

- **--dataset**: Path to the dataset for learning.
- **--model**: Path to save the fine-tuned CNN model.

## Environment

Developed using Python 3, scikit-learn 0.19, Keras 2.1, and OpenCV 3 on an NVIDIA RTX 3050.

## Results

Validation accuracy:
- **Warm-Up Phase (25 epochs)**: 82.35%
- **Final Phase (100 epochs)** with unfreezing Conv layers 15 onwards: 95.59%

Classification report using sklearn for 17 classes:
- **Precision**: Ability to avoid labeling negative samples as positive.
- **Recall**: Ability to find all positive samples.
- **Support**: Number of occurrences of each class in y_true.

