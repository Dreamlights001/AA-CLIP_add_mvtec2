# 复现AA-CLIP并添加MVTec2数据集支持计划

## 任务目标
1. 修改数据集路径配置，将所有数据集路径指向 `/root/autodl-tmp/datasets`
2. 添加MVTec2数据集支持
3. 创建MVTec2数据集的metadata文件

## 实施步骤

### 1. 修改数据集路径配置
- 编辑 `dataset/constants.py` 文件
- 将 `BASE_PATH` 从 `/data/wenxinma` 修改为 `/root/autodl-tmp/datasets`

### 2. 添加MVTec2数据集配置
- 在 `DATA_PATH` 字典中添加 `"MVTec2": f"{BASE_PATH}/MVTec2"`
- 在 `CLASS_NAMES` 字典中添加MVTec2的类别列表：
  ```python
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
  ```
- 在 `DOMAINS` 字典中添加 `"MVTec2": "Industrial"`
- 在 `REAL_NAMES` 字典中为MVTec2的每个类别添加描述

### 3. 创建MVTec2数据集的metadata文件
- 在 `dataset/metadata/` 目录下创建 `MVTec2` 文件夹
- 创建 `full-shot.jsonl` 文件，按照以下格式生成条目：
  ```json
  {"image_path": "can/test_public/bad/xxx.png", "label": 1.0, "mask_path": "can/test_public/ground_truth/xxx.png", "class_name": "can"}
  {"image_path": "can/test_public/good/xxx.png", "label": 0.0, "mask_path": "", "class_name": "can"}
  ```
- 为每个类别（can, fabric, fruit_jelly, rice, sheet_metal, vial, wallplugs, walnuts）生成对应的条目

### 4. 验证配置
- 确保所有数据集路径正确映射到 `/root/autodl-tmp/datasets` 下的对应位置
- 确保MVTec2数据集的metadata文件格式正确，路径指向正确

## 预期结果
- 项目能够使用新的数据集路径配置加载所有数据集
- 项目能够正确识别和加载MVTec2数据集
- 能够成功运行训练和测试脚本