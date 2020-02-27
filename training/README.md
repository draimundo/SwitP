# SwitP/training
Folder containing data related to the training of the Neural Network.

The .yml files in the root can be used to replicate the conda environment used, by using conda env create -f=/path/to/environment.yml in the conda prompt.
The SwitPgpu training adds support for compatible nvidia GPUs, tested on Windows 10 with NVIDIA GPU driver 442.19, CUDA Toolkit 10.0.130 and CUDANN 7.6.5.0. Details under https://www.tensorflow.org/install/gpu