# Infrastructure-Assisted Cooperative Decision Model With Priority Awareness at Unsignalized Intersections
This repository contains the code used for the experiments in our paper of the above name, along with experimental results  of  (code will be released soon!)

## 🚀 Main Contributions
Our contributions can be mainly divided into the following parts:

**1. Vehicle-to-Infrastructure Integrated Distributed Agent Decision-Making Framework:** This study proposes a Vehicle-to-Infrastructure Integrated Distributed Agent Decision-Making (V2I-IDADM) framework for unsignalized intersections, in which roadside infrastructure plays a central role in aggregating global traffic information and coordinating multi-vehicle behaviors. By shifting global coordination from individual vehicles to the infrastructure, the proposed framework establishes a scalable and safety-critical cooperative decision-making paradigm. 

**2.model-driven cooperative decision-making model:** A model-driven cooperative decision-making model is developed for intersection scenarios, which combines priority-aware safety optimization with hierarchical weighted sampling. By assigning passing priorities, optimizing global vehicle actions, and emphasizing informative high-reward samples during training, the model improves both traffic safety and efficiency while enhancing the self-learning capability of CAVs.

**3.Simulation and Real-World Validation:** Extensive experiments are conducted under both pure CAV and mixed-traffic environments involving human-driven vehicles, demonstrating the robustness and effectiveness of the proposed framework. Furthermore, the miniature intelligent vehicle and the full-scale vehicle experiments are developed to validate the practical feasibility and real-world deployment potential of the proposed V2I-IDADM framework.

# 💥 Miniature intelligent vehicles testing
A miniature intelligent vehicle testbed is constructed to emulate an intersection scenario. Four vehicles are deployed at different approaches, while a motion capture system provides global traffic information such as position, velocity, and heading. Each vehicle communicates with a MEC server through Wi-Fi. The MEC server runs the PCSQMIX-HWS model to optimize cooperative decisions and sends the resulting commands to the vehicles for execution.

<div align="center">
  <img width="1200" alt="image" src="https://github.com/user-attachments/assets/bfe677c8-ebc2-422f-a6a7-81dea291d896" />
</div>

https://github.com/user-attachments/assets/83e60f4a-c43f-4912-aa49-0846cc3e7a86


# 💥 Real-Vehicle testing
a full-scale vehicle experiment is conducted in a four-vehicle cooperative unsignalized intersection scenario. Each CAV is equipped with a GNSS/INS navigation system and a 5G communication gateway, while the RSU also uses a 5G gateway for V2I communication. The vehicles’ position, velocity, and heading information are collected through GNSS/INS and transmitted to the MEC server via 5G. The MEC server runs the PCSQMIX-HWS model to generate cooperative decisions and sends the optimized commands back to the CAVs for execution.

<div align="center">
  <img width="1200" alt="image" src="https://github.com/user-attachments/assets/672fa390-51c3-4551-bfdb-9665bbbc6f42"/>
</div>

https://github.com/user-attachments/assets/bde1734d-9196-41b2-887b-ef04d1054441
