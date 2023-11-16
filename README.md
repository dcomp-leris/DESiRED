# DESiRED# Purpose of this repository
This repository aims to make the artifacts used in article "DESiRED - Dynamic, Enhanced, and Smart iRED: A P4-AQM with Deep Reinforcement Learning and In-band Network Telemetry (Computer Networks - Submitted)" available.

# Definition
DESiRED is a joint solution that leverages the In-band Network Telemetry and Deep Reinforcement Learning to adjust the AQM target delay in real-time.  With DESiRED, fine-grained measurements are used as an input layer for the Deep Queue Network to define management actions, that is, find the ideal target delay based on the current network conditions. Based on the experience of the actions taken, DESiRED learns the ideal target delay values following a rewards policy that maximizes the QoS. With DESiRED dynamically adjusting the target delay, we achieved the best results compared to a fixed target delay.

# Design of iRED
![alt-text](https://github.com/dcomp-leris/iRED_TNSM/blob/main/figs/iRED.jpg)

# Folders
```console
.
├── evaluation # files used in the evaluation section
├── figs # figures
├── README.md
└── source-code # DESiRED source code 
```

