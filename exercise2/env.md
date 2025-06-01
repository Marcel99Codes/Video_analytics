# Video_analytics - Exercise 2

## Installation:
conda install -r requirements.txt

## Task 1
* Change the datapath in `main.py`:  
  `DATA_PATH = "/media/marcel/Data1/video_analytics/data"`
* Training of all models (rgb, flow-random, flow-imagenet):  
  `python3 main.py --train`
* Training of a specific model:  
  `python3 main.py --train --model rgb`
* Validation of all models (rgb, flow-random, flow-imagenet):  
  `python3 main.py --validate`
* Validation of a specific model:  
  `python3 main.py --validate --model rgb`
* Validation of the fusion of rgb and flow-random:  
  `python3 main.py --fusion`

## Task 2
* Change the datapath in `main.py`:  
  `DATA_PATH = "/media/marcel/Data1/video_analytics/data"`
* Training with random initialization:  
  `python3 main.py --init random`
* Training with ImageNet initialization:  
  `python3 main.py --init imagenet`

