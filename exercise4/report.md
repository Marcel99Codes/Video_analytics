### Report Exercise4: 

* Subject: Video Analytics
* Name: Marcel Plocher


## Step1: Augmentation
In the following you can see some visulized examples: 

# Data augmentation strategy

1. Take the frames from the video tensors [C, T, H, W] ==> each frame [C, H, W]
2. Per frame brightness jittering: Randomly adjust the brightness of each frame t
 * Simulate light changes
 * Incorperate ilimmination invariance
3. Per frame horizonal flip: Flip the frames with a propability of 50% 
 * Change spatial orientation
 * Incorperate viewpoint invariance 
4. Per frame gaussian smooth blur: Remove high-frequency content to make the image more soft
 * Better generalization and preventing overfitting 
 * Encouraging robustness 
5. Stack the frames back along the time axis to reconstract the transformed clip


## Step2: Constrastive Learning Setup: 

# Implementation NT-Xent loss
# Triplet loss
# InfoNCE loss 

## Step3: Pretraining the Model

# Training loss:

# Model weights:


## Step4: Finetuning for Action Recognition:

# Final accuracy


## Running Code Guidline:
