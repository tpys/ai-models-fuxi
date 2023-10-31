# ai-models-fuxi

`ai-models-fuxi` is an [ai-models](https://github.com/ecmwf-lab/ai-models) plugin to run [Fudan's FuXi](https://github.com/tpys/FuXi.git).

FuXi: A cascade machine learning forecasting system for 15-day global weather forecast,arXiv preprint: 2306.12873, 2022. https://arxiv.org/pdf/2306.12873.pdf

FuXi was created by Lei Chen, Xiaohui Zhong, Feng Zhang, Yuan Cheng, Yinghui Xu, Yuan Qi, Hao Li. It is released by Fudan University.

The model weights are made available under the terms of the BY-NC-SA 4.0 license.
The commercial use of these models is forbidden.

See <https://github.com/tpys/FuXi.git> for further details.


### Installation

To install the package, run:

```bash
pip install ai-models-fuxi
```

This will install the package and its dependencies, in particular the ONNX runtime. The installation script will attempt to guess which runtime to install. You can force a given runtime by specifying the the `ONNXRUNTIME` variable, e.g.:

```bash
ONNXRUNTIME=onnxruntime-gpu pip install ai-models-fuxi
```
