# IOAI Chicken Counting (Density Estimation)

## Solution Overview
This repository contains my solution to the IOAI Chicken Counting task (crowd counting via density estimation).

I implemented a UNet architecture from scratch in PyTorch to predict density maps and estimate the number of chickens in each image.

Final score: **0.87**

---

## Approach
- Implemented custom UNet (encoder-decoder architecture)
- Used density estimation instead of direct counting
- Designed model to output 180×320 density maps
- Optimized training pipeline for better generalization

The total number of chickens is obtained by summing the predicted density map.

---

## 1. Problem Description
As the leader of an AI research team collaborating with Silkie chicken farmers, you are tasked with solving a critical challenge in traditional free-range farming. Accurate counting of livestock is crucial for both farmers and insurance companies, as factors like disease outbreaks and predator invasions can significantly impact the survival rate of these chickens in a short time. While insurance coverage helps mitigate farming risks, the claims process requires precise counting of livestock losses. Your farmers have approached your team for help in developing more accurate, automated counting systems. The challenge before your research team is to develop an optimized Silkie chicken counting model using density estimation techniques that can provide reliable counts to support both farm management and insurance processes.

Your team has access to a pretrained feature extractor for Silkie chicken images, but you'll need to design and train the density estimation decoder to create a complete counting solution. Your task is to build upon this foundation by developing an effective decoder architecture and training strategy to achieve accurate chicken counts that farmers and insurance companies can rely on.

---

## 2. Dataset
The structure of the provided Silkie chicken image dataset is as follows:

datasets/
├── train/
│   └── A dataset with features:
│       ├── `image`: `PIL.Image` with RGB channels (3x720x1280)
│       └── `density`: a 2D array of shape 180x320
└── base.pth (Pretrained Model)

os.environ.get("DATA_PATH")/
├── test_a/
│   └── A dataset with features:
│       └── `image`: `PIL.Image` with RGB channels (3x720x1280)
└── test_b/
    └── A dataset with features:
        └── `image`: `PIL.Image` with RGB channels (3x720x1280)

Training set: 100 images  
Validation set: 100 images  
Test set: 100 images  

Density maps are shaped as **1×180×320**, where the sum of values equals the number of chickens.

---

## 3. Task
Train a model to predict density maps from input images.

- Input: image (3×720×1280)  
- Output: density map (180×320)  

The predicted count is the sum of all values in the density map.

---

## 4. Submission
Submit a `submission.ipynb` containing:

- Training pipeline  
- Evaluation code  

Output: `submission.npz` with:
- `pred_a` (validation predictions)  
- `pred_b` (test predictions)  

Each with shape:
- 100×1×180×320 or 100×180×320  

---

## 5. Scoring
Metric: Mean Relative Error

Relative Error:
|yi − ŷi| / |yi|

Final score:
exp(−mean relative error)

---

## 6. Baseline & Results
- Baseline score: 0.71  
- Best (committee): 0.89  
- My score: **0.87**
