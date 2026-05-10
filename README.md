# Infrastructure-Assisted Cooperative Decision Model With Priority Awareness at Unsignalized Intersections
This repository contains the code used for the experiments in our paper of the above name, along with experimental results  of  (code will be released soon!)

# Main Contributions
Our contributions can be mainly divided into the following parts:

1.Vehicle-to-Infrastructure Integrated Distributed Agent Decision-Making Framework： This study proposes a Vehicle-to-Infrastructure Integrated Distributed Agent Decision-Making (V2I-IDADM) framework for unsignalized intersections, in which roadside infrastructure plays a central role in aggregating global traffic information and coordinating multi-vehicle behaviors. By shifting global coordination from individual vehicles to the infrastructure, the proposed framework establishes a scalable and safety-critical cooperative decision-making paradigm. 
2.Multi-metric output and event reasoning for road traffic: 
3.Simulation and Real-World Validation: Extensive experiments are conducted under both pure CAV and mixed-traffic environments involving human-driven vehicles, demonstrating the robustness and effectiveness of the proposed framework. Furthermore, the miniature intelligent vehicle and the full-scale vehicle experiments are developed to validate the practical feasibility and real-world deployment potential of the proposed V2I-IDADM framework.

# Miniature intelligent vehicles testing
https://github.com/user-attachments/assets/83e60f4a-c43f-4912-aa49-0846cc3e7a86

# Real-Vehicle testing
https://github.com/user-attachments/assets/be8f6e8f-af82-4cf6-b9bb-40964c799ef4

# Requirements and Installation
gym

highway_env

numpy

tensorflow

tf_slim

torch

# Running the Code
Before running the code, you need to create two folders, model and result, to store the model parameters and experimental results respectively.

# Training
To train the model, use the following command:

python main.py
