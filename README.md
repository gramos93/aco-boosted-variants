# An Ant Colony with Visibility Problem optimization approach for Multi-UAV based, search and rescue missions during forest wildfires.

<p align="center">
<img src="../radioactive-goose/assets/radioactive-goose-logo.jpg" alt="logo" width="300"/>
</p>

This repository is the official implementation of [An Ant Colony with Visibility Problem optimization approach for Multi-UAV based, search and rescue missions during forest wildfires.](https://arxiv.org/abs/2030.12345).

### Authors
- **Department of Computer Science, Université Laval :**

    - Gabriel Ramos
    - Kevin Laurent 
    - Felix Méthot
    - Gabriel Jeanson

### Abstract

> This paper introduces an innovative Ant Colony with Visibility Problem (ACVP) optimization approach tailored for coordinating multi-agent drone-based teams during operations in search and rescue missions amid forest wildfires. Leveraging the collective intelligence of the Ant-Colony algorithm and a visibility heuristic, this algorithm orchestrates multiple UAVs to navigate hazardous wildfire environments efficiently. Through simulations and empirical evaluations, the ACVP algorithm demonstrates significant improvements in search time, area coverage, and victim detection rates compared to traditional methods. This approach offers a promising solution for real-time implementation in disaster management systems, providing an effective tool for responders to enhance search and rescue operations and mitigate the impact of forest wildfires in challenging conditions.

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Cite

```

```

