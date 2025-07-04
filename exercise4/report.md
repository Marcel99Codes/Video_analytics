### Report Exercise4: 

* Subject: Video Analytics
* Name: Marcel Plocher

## Running Code Guidline:

 * Install the requirements.txt: 
   * *pip install -r requirements.txt*
 * Adapt the root path to the mini_UCF dataset in main.py (line 14)
   * DATA_ROOT_PATH = "/media/marcel/Data1/video_analytics/data/mini_UCF"
 * Visualize a data augumentation sample: 
     * *python main.py --visualize* 
 * Start pretraining :
     * *python main.py --pretrain*
 * Start finetuning:
     * *python main.py --finetune*
 * You can also adapt the hyperparameters (lr, batch_size, epochs)
     * *python main.py --pretrain --batch_size=8*


## Step1: Data Augmentation

In the first step the task was to find a good data augumentation pipeline for self-supervied learning. I have choosen a effective but also lightweight pipeline. In the following I will discuss the single steps of the strategy: 

1. Take the frames from the video tensors [C, T, H, W] ==> each frame [C, H, W]
2. Per frame brightness jittering: Randomly adjust the brightness of each frame t
   * Simulate light changes
   * Incorperate illumination invariance
3. Per frame horizonal flip: Flip the frames with a propability of 50% 
   * Change spatial orientation
   * Incorperate viewpoint invariance 
4. Per frame gaussian smooth blur: Remove high-frequency content to make the image more soft
   * Better generalization and preventing overfitting 
   * Encouraging robustness 
5. Stack the frames back along the time axis to reconstract the transformed clip

In the following you can see some visulized examples: <br>

![Alt text](images/Fig2.png)
![Alt text](images/Fig3.png)

## Step2: Constrastive Learning Setup: 

After defining the augumentation pipeline the next task was to implement diffrent constractive losses. For this i made a own file *losses.py*. You also find the implementation in the following:

### Implementation NT-Xent loss
```
def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2N x feature_dim
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature  # 2N x 2N similarity matrix
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)

    targets = torch.cat([torch.arange(batch_size, 2*batch_size),
                         torch.arange(0, batch_size)]).to(z.device)

    loss = F.cross_entropy(sim, targets)
    return loss
```

### Implementation Triplet loss
```
def triplet_loss(anchor, positive, negative, margin=0.2):
    pos_dist = F.l1_loss(anchor, positive, reduction='none').mean(1)
    neg_dist = F.l1_loss(anchor, negative, reduction='none').mean(1)
    return F.relu(pos_dist - neg_dist + margin).mean()
```

### Implementation InfoNCE loss
```
def info_nce_loss(z_anchor, z_positive, temperature=0.5):
    batch_size = z_anchor.shape[0]
    z_anchor = F.normalize(z_anchor, dim=1)
    z_positive = F.normalize(z_positive, dim=1)
    logits = torch.mm(z_anchor, z_positive.t()) / temperature
    labels = torch.arange(batch_size).to(z_anchor.device)
    return F.cross_entropy(logits, labels)
```

## Step3: Pretraining the Model
In step 3 we want to train 
### Training loss:

### Model weights:


## Step4: Finetuning for Action Recognition:

### Final accuracy

