# MVTec2 数据集说明

## 数据集简介

MVTec2 是一个用于异常检测的工业数据集，包含了多种工业产品的正常和异常样本。该数据集扩展了原始的 MVTec AD 数据集，增加了更多的样本和类别。

## 目录结构

MVTec2 数据集的目录结构如下：

```
MVTec2
├── can
│   ├── test_private
│   ├── test_private_mixed
│   ├── test_public
│   │   ├── bad
│   │   ├── good
│   │   └── ground_truth
│   ├── train
│   │   └── good
│   └── validation
├── fabric
├── fruit_jelly
├── rice
├── sheet_metal
├── vial
├── wallplugs
├── walnuts
├── license.txt
└── readme.txt
```

## 类别说明

MVTec2 数据集包含以下类别：

| 类别 | 描述 |
|------|------|
| can | 金属罐 |
| fabric | 织物纹理 |
| fruit_jelly | 果冻 |
| rice | 米粒 |
| sheet_metal | 金属板材 |
| vial | 小玻璃瓶 |
| wallplugs | 墙塞 |
| walnuts | 核桃 |

## 文件组织

### 训练集

训练集位于每个类别的 `train/good` 目录下，只包含正常样本。

### 测试集

测试集位于每个类别的 `test_public` 目录下，包含：

- `bad` 目录：包含异常样本
- `good` 目录：包含正常样本
- `ground_truth` 目录：包含异常样本的掩码文件

### 私有测试集

私有测试集位于每个类别的 `test_private` 和 `test_private_mixed` 目录下，用于模型评估。

## 文件命名约定

MVTec2 数据集的文件命名包含了位姿变换信息，例如：

- `000_overexposed.png`：过曝光样本
- `000_regular.png`：正常曝光样本
- `000_shift_1.png`：位移变换样本
- `000_shift_2.png`：位移变换样本
- `000_shift_3.png`：位移变换样本
- `000_underexposed.png`：曝光不足样本

## 在 AA-CLIP 中的使用

### 数据集配置

在 `dataset/constants.py` 文件中，MVTec2 数据集的配置如下：

```python
DATA_PATH = {
    # 其他数据集...
    "MVTec2": f"{BASE_PATH}/mvtec2",
}

CLASS_NAMES = {
    # 其他数据集...
    "MVTec2": [
        "can",
        "fabric",
        "fruit_jelly",
        "rice",
        "sheet_metal",
        "vial",
        "wallplugs",
        "walnuts"
    ],
}

DOMAINS = {
    # 其他数据集...
    "MVTec2": "Industrial",
}

REAL_NAMES = {
    # 其他数据集...
    "MVTec2": {
        "can": "metal can",
        "fabric": "fabric texture",
        "fruit_jelly": "fruit jelly",
        "rice": "rice grains",
        "sheet_metal": "sheet metal",
        "vial": "small glass vial",
        "wallplugs": "wall plugs",
        "walnuts": "walnuts"
    },
}
```

### 元数据文件

MVTec2 数据集的元数据文件位于 `dataset/metadata/MVTec2/full-shot.jsonl`，包含了所有样本的路径和标签信息。

### 测试命令

要测试 MVTec2 数据集，可以使用以下命令：

```bash
python test.py --save_path ./ckpt --dataset MVTec2
```

## 异常检测任务

MVTec2 数据集可用于以下异常检测任务：

1. **图像级异常检测**：判断整个图像是否包含异常
2. **像素级异常检测**：定位图像中异常的具体位置
3. **少样本异常检测**：使用少量正常样本进行模型训练
4. **零样本异常检测**：不使用目标数据集的样本进行模型训练

## 数据集特点

1. **多样性**：包含多种工业产品类别
2. **丰富的异常类型**：每个类别包含多种异常类型
3. **位姿变换**：样本包含多种位姿和光照条件的变换
4. **精确的掩码标注**：异常区域有精确的像素级标注
5. **标准化的目录结构**：与原始 MVTec AD 数据集保持一致的目录结构

## 注意事项

1. **文件路径**：在使用数据集时，需要确保代码能够正确处理带有位姿变换后缀的文件名
2. **掩码文件**：掩码文件可能使用不同的命名约定，需要代码能够灵活适配
3. **目录结构**：不同类别的目录结构可能略有差异，需要代码能够适应不同的目录结构
4. **数据加载**：由于数据集包含大量样本，建议使用批量加载和数据增强技术

