---
Title: Install TensorFlow and PyTorch Together in a Single Environment
Date: 2021-08-29 23:39:42
Category: Machine Learning
---

I have been recently dealing with projects requiring both the popular automatic differentiation frameworks together.
Installing both TensorFlow and PyTorch in a single environment might be a challenge of its own kind.

I'm, here, writing down the recipe I figured out to get these two frameworks installed.
As we later show, both libraries can have access to GPU from a single Python interpreter instance.


### Specifications
The OS and library versions are as follows:
* Ubuntu 20.04.3
* Python 3.8.11
* Tensorflow 2.4.1 (`tensorflow-gpu`)
* PyTorch 1.9.0


### Installation
First, let's create a Conda environment.
I choose to install TensorFlow along with the creation of the environment.
Having a GPU installed on the machine, so I opt for the GPU-ready binaries `tensorflow-gpu`.
Conda makes your life much easier by handling the installation of the right binaries compatible with your CUDA device(s):
```shell
$ conda create --name tfpt8 python=3.8 tensorflow-gpu
```

Once the env is created, you need to activate it by:
```shell
$ conda activate tfpt8
```

Then we install PyTorch. PyPI can do the job with less clutter:
```shell
$ pip install torch
```


### Sanity check
Now it's time to test the recipe!
Let's run an interpreter instance to see if everything is working properly:
```python
Python 3.8.11 (default, Aug  3 2021, 15:09:35) 
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
2021-08-30 00:17:34.532729: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.10.1
>>> import torch
>>> 
```
Both libraries are imported correctly.
We want to find out more about GPU access:
```python
>>> tf.config.list_physical_devices('GPU')
2021-08-30 00:25:16.374593: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-08-30 00:25:16.375472: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcuda.so.1
2021-08-30 00:25:16.401750: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce RTX 2060 SUPER computeCapability: 7.5
coreClock: 1.68GHz coreCount: 34 deviceMemorySize: 7.76GiB deviceMemoryBandwidth: 417.29GiB/s
2021-08-30 00:25:16.401779: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudart.so.10.1
2021-08-30 00:25:16.403444: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.10
2021-08-30 00:25:16.403481: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublasLt.so.10
2021-08-30 00:25:16.405139: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcufft.so.10
2021-08-30 00:25:16.405372: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcurand.so.10
2021-08-30 00:25:16.406924: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusolver.so.10
2021-08-30 00:25:16.407862: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcusparse.so.10
2021-08-30 00:25:16.411216: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcudnn.so.7
2021-08-30 00:25:16.411943: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

```python
>>> torch.cuda.is_available()
True
>>> torch.cuda.get_device_name(0)
'GeForce RTX 2060 SUPER'
```

Voilà!! As you can see both `tensorflow` and `torch` have recognized the GPU on the machine.

