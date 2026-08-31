# Domain Shift-Robust Multi-Organ Segmentation via Random Convolution-Intervened Causal Modeling

## Overview

![overview]

## Getting Started

Multi-organ segmentation model

### Prerequisites

Our modeling framework is based on PyTorch (https://pytorch.org).

install `nnUNet`：  
```
pip install nnUNet
```
Install the source code of the nnUNet model:
```
git clone https://github.com/MIC-DKFZ/batchgeneratorsv2.git
```
```
Cd batchgeneratorsv2
```
```
Pip install -e .
```
In the same way, install the network architecture to build your own model:
```
git clone https://github.com/MIC-DKFZ/dynamic-network-architectures.git
Cd dynamic-network-architectures
Pip install -e .
```
Find `nnUNetPlans.json` to use your own model:
```
"architecture": {
                "network_class_name": "dynamic_network_architectures.architectures.mynet.MyNet",
...
}
```


## Running the Model

### Model

- Run model with `./Code/R/xxx`
- Run the xx methods:
  - xx：`./Code/R/xxx`
  - xxx：`./Code/xxR`


### Analysis



## Authors

* Huijun Li, 


## License

This project is licensed under the BSD 3-Clause License, see [LICENSE](LICENSE) details.

## Acknowledging this work

If you publish any work based on the contents of this repository please cite:




