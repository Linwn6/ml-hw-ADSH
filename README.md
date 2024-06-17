# ml-hw-ADSH


# ADSH: 非对称深度监督哈希

该仓库包含了[非对称深度监督哈希](https://cs.nju.edu.cn/lwj/paper/AAAI18_ADSH.pdf)（ADSH）算法的实现。实现基于[ADSH_PyTorch](https://github.com/TreezzZ/ADSH_PyTorch)、[ADSH matlab + pytorch](https://github.com/jiangqy/ADSH-AAAI2018)和[DeepHash-pytorch](https://github.com/swuxyj/DeepHash-pytorch)的代码。

## 目录

- [安装](#安装)
- [数据集](#数据集)
- [运行代码](#运行代码)
- [参数设置](#参数设置)
- [精确召回曲线](#精确召回曲线)
- [模型下载](#模型下载)
- [引用](#引用)
- [致谢](#致谢)
  

## 安装

### GPU运行环境参考
- Python 3.12.0
- PyTorch 1.10.0+cu113
- torchvision 0.11.0+cu113

### CPU运行环境参考
- Python 3.12
- PyTorch 2.3.1
- torchvision 0.18.1

### 安装步骤

1. 克隆仓库：
    ```bash
    git clone (https://github.com/Linwn6/ml-hw-ADSH)
    cd ADSH
    ```

2. 安装所需的包
   

## 数据集

### NUSWIDE-10
- **子集**：包含10个最受欢迎的类。
- **查询集**：随机选择5000张图像。
- **检索集**：其余图像。
- **训练集**：从检索集中随机选择10,500张图像。

### CIFAR-10
- **描述**：包含来自10个类的60,000张真实图像。
- **查询集**：每类随机选择1,000张图像。
- **检索集**：其余图像。
- **训练集**：从检索集中每类随机选择500张图像（共5,000张）。

## 运行代码

要运行ADSH算法的主文件，请执行：

```bash
python ADSH.py
```

## 参数设置

ADSH算法的配置如下：

```python
def get_config():
    config = {
        "gamma": 200,  # 量化损失系数
        "num_samples": 2000,  # 每次迭代的样本数量
        "max_iter": 150,  # 最大迭代次数
        "epoch": 3,  # 每次迭代的训练轮数
        "test_map": 10,  # 测试间隔
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-4, "weight_decay": 1e-5}},  # 优化器类型和参数
        "info": "[ADSH]",
        "resize_size": 256,  # 图像调整大小
        "crop_size": 224,
        "batch_size": 64,  # 批处理大小
        "net": AlexNet,  # 神经网络模型
        "dataset": "nuswide_21",  # 数据集名称
        "save_path": "save/ADSH",  # 模型保存路径
        "device": torch.device("cuda:0"),  # 设备（CPU或GPU）
        "bit_list": [48],  # 哈希位数列表
    }
    if config["dataset"] == "nuswide_21":
        config["gamma"] = 0
    config = config_dataset(config)
    return config
```

## 精确召回曲线

生成精确召回曲线的步骤如下：

1. 将JSON路径 `"ADSH": "../log/alexnet/ADSH_cifar10-1_48.json"` 复制到 `precision_recall_curve.py`。

1. 运行脚本生成曲线：

    ```bash
    cd utils
    python precision_recall_curve.py
    ```
## 模型下载

可直接在releases的save_models下载已训练好的模型解压使用。

## 引用

如果你觉得这个仓库对你有帮助，请考虑引用：

```
@article{jiang2018asymmetric,
  title={Asymmetric deep supervised hashing},
  author={Jiang, Qin-Yu and Li, Wei-Jie},
  journal={AAAI},
  volume={333},
  pages={334},
  year={2018}
}
```

## 致谢

此代码基于以下仓库：
- [ADSH matlab + pytorch](https://github.com/jiangqy/ADSH-AAAI2018)
- [ADSH_PyTorch](https://github.com/TreezzZ/ADSH_PyTorch)
- [DeepHash-pytorch](https://github.com/swuxyj/DeepHash-pytorch)
---
