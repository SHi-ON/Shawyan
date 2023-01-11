---
title: "Get Up and Running with NVIDIA GPUs on Ubuntu"
date: 2023-01-10T21:42:42-04:00
draft: false
author: SHi-ON
---

I have been setting up multiple Python machine learning environments based on NVIDIA GeForce GPUs. Starting with a fresh Ubuntu installation, you will need to go through the installation of multiple NVIDIA packages to get your machine ready to communicate with the GPU(s) properly. The steps here are an overview of the installation process.

### Specifications
Here are my system specifications:
 - Ubuntu 22.04 LTS 
 - NVIDIA GeForce RTX 3090 x2


## Installation
Since drivers and toolkits follow different versioning schemes, I have added the latest versions as well just to make it easier to know what is what!

### 1. NVIDIA Drivers

 - Versioning scheme: XXX.XX.XX
 - [Latest version](https://www.nvidia.com/en-us/drivers/unix/) as of today: 525.78.01

It is recommended to install the drivers via the OS package manager. You can follow the steps [here](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html#ubuntu-lts).


### 2. CUDA Toolkit

 - Versioning scheme: XX.X.X
 - [Latest version](https://developer.nvidia.com/cuda-toolkit-archive) as of today: 12.0.0

You can find the official CUDA Toolkit installation instructions on Linux [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). One-time skimming through the entire documentation is recommended as there are a variety of dependencies and settings depending on your OS and the expected use case. In a high-level view, the process at this stage can be broken down into three main steps: 1. pre-installation actions, 2. main installation, and 3. post-installation actions

#### 2.1 Pre-Installation Actions
As long as you are having a typical setup, there is no special step to take [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions).

#### 2.2 Package Manager Installation
I prefer the Network Repo (over the Local Repo) installation as it gives you the ability to update through the APT package manager in the future. Follow the instructions for Ubuntu [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#network-repo-installation-for-ubuntu). In summary, you need to install the `cuda-keyring` and then the CUDA SDK. Once you finish, you need to reboot your system.

#### 2.3 Post-Installation Actions
Some finishing touches can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions). Make sure you have updated the `PATH` variable. Adding the `export` shell command to your shell configuration (e.g. bashrc or zshrc) file is recommended to keep the `PATH` variable consistent throughout a shell session.


### 3. cuDNN

 - Versioning scheme: X.X.X
 - [Latest version](https://docs.nvidia.com/deeplearning/cudnn/archives/index.html) as of today: 8.7.0

The steps here are similar to the ones from the previous step. Install the package according to the documentation [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-linux).

## Finale
Once you have all the packages installed, you can verify your installation. In the case of machine learning you can follow the quick checks with the PyTorch and TensorFlow libraries from [my other post](https://shayanamani.com/posts/tensorflow_pytorch_env/#sanity-check).