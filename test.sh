#!/bin/bash

# 创建 results 目录
mkdir -p ./results

# 设置 checkpoint 路径
save_path="./ckpt"

# 检查 checkpoint 目录是否存在
if [ ! -d "$save_path" ]; then
    echo "Error: Checkpoint directory $save_path does not exist!"
    echo "Please train the model first before testing."
    echo "Training command example: python train.py --save_path $save_path --dataset MVTec2 --shot 0"
    exit 1
fi

# 检查是否存在 image adapter checkpoint 文件
if [ -z "$(ls -la $save_path | grep image_adapter_)" ]; then
    echo "Error: Image adapter checkpoint files not found in $save_path!"
    echo "Please train the model first before testing."
    echo "Training command example: python train.py --save_path $save_path --dataset MVTec2 --shot 0"
    exit 1
fi

# 定义数据集数组
declare -a dataset=(MVTec BTAD MPDD Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir Colon_cvc300 VisA MVTec2)

# 循环测试每个数据集
for i in "${dataset[@]}"; do
    echo "Testing $i..."
    # 运行测试命令
    python test.py --save_path $save_path --dataset $i
    # 将测试结果复制到 results 目录
    if [ -f "$save_path/test.log" ]; then
        cp "$save_path/test.log" "./results/${i}_test.log"
        echo "Test results for $i saved to ./results/${i}_test.log"
    fi
done

echo "All tests completed!"
