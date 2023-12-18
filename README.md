# An Ant Colony with Visibility Problem optimization approach for Multi-UAV based, search and rescue missions during forest wildfires.

### Authors
- **Department of Computer Science, Université Laval :**

    - Gabriel Ramos
    - Kevin Laurent 
    - Felix Méthot
    - Gabriel Jeanson

### Abstract

> This paper introduces an innovative Ant Colony with Visibility Problem (ACVP) optimization approach tailored for coordinating multi-agent drone-based teams during operations in search and rescue missions amid forest wildfires. Leveraging the collective intelligence of the Ant-Colony algorithm and a visibility heuristic, this algorithm orchestrates multiple UAVs to navigate hazardous wildfire environments efficiently. Through simulations and empirical evaluations, the ACVP algorithm demonstrates significant improvements in search time, area coverage, and victim detection rates compared to traditional methods. This approach offers a promising solution for real-time implementation in disaster management systems, providing an effective tool for responders to enhance search and rescue operations and mitigate the impact of forest wildfires in challenging conditions.

## Requirements

To install requirements:

```setup
pip install numpy
```

## Training

To start the model run this command:

```train
python src/core.py
```

To evaluate the models, run:

```eval
python src/test.py
```
