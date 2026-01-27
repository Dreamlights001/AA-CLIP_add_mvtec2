# AA-CLIP: Enhancing Zero-shot Anomaly Detection via Anomaly-Aware CLIP
 **[CVPR 2025 paper]**

[![Paper](https://img.shields.io/badge/CVPR-Paper-red)](https://arxiv.org/pdf/2503.06661) [![Appendix](https://img.shields.io/badge/CVPR-Appendix-blue)](https://drive.google.com/file/d/1PQrjCvWDyuM7W2ClJ-cJeD4YKJ1uPAzc/view?usp=drive_link)
 Official Pytorch Implementation

![](pic/teaser.png)

## Abstract
Anomaly detection (AD) identifies outliers for applications like defect and lesion detection. While CLIP shows promise for zero-shot AD tasks due to its strong generalization capabilities, its inherent **Anomaly-Unawareness** leads to limited discrimination between normal and abnormal features. To address this problem, we propose **Anomaly-Aware CLIP** (AA-CLIP), which enhances CLIP's anomaly discrimination ability in both text and visual spaces while preserving its generalization capability. AA-CLIP is achieved through a straightforward yet effective two-stage approach: it first creates anomaly-aware text anchors to differentiate normal and abnormal semantics clearly, then aligns patch-level visual features with these anchors for precise anomaly localization. This two-stage strategy, with the help of residual adapters, gradually adapts CLIP in a controlled manner, achieving effective AD while maintaining CLIP's class knowledge. Extensive experiments validate AA-CLIP as a resource-efficient solution for zero-shot AD tasks, achieving state-of-the-art results in industrial and medical applications. 

## Results
![](pic/results.png)

## Quick Start 
### 1. Installation  
```bash
git clone https://github.com/Mwxinnn/AA-CLIP.git
cd AA-CLIP
conda create -n aaclip python=3.10 -y  
conda activate aaclip  
pip install -r requirements.txt  
```
### 2. Datasets
The datasets can be downloaded from [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/), [VisA](https://github.com/amazon-science/spot-diff), [MPDD](https://github.com/stepanje/MPDD), [BrainMRI, LiverCT, Retinafrom](https://drive.google.com/drive/folders/1La5H_3tqWioPmGN04DM1vdl3rbcBez62?usp=sharing) from [BMAD](https://github.com/DorisBao/BMAD), [CVC-ColonDB, CVC-ClinicDB, Kvasir, CVC-300](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579) from Polyp Dataset.

Put all the datasets under ``./data`` and use jsonl files in ``./dataset/metadata/``. You can use your own dataset and generate personalized jsonl files with below format:
```json
{"image_path": "xxxx/xxxx/xxx.png", 
 "label": 1.0 (for anomaly) # or 0.0 (for normal), 
 "class_name": "xxxx", 
 "mask_path": "xxxx/xxxx/xxx.png"}
```
The way of creating corresponding jsonl file differs, depending on the file structure of the original dataset. The basic logic is recording the path of every image file, mask file, anomaly label ``label``, and category name, then putting them together under a jsonl file.

(Optional) the base data directory can be edited in ``./dataset/constants``. If you want to reproduce the results with few-shot training, you can generate corresponding jsonl files and put them in ``./dataset/metadata/{$dataset}`` with ``{$shot}-shot.jsonl`` as the file name. For few-shot training, we use ``$shot`` samples from each category to train the model.

> Notice: Since the anomaly scenarios in VisA are closer to real situations, the default hyper-parameters are set according to the results trained on VisA. More analysis and discussion will be available.

### 3. Training & Evaluation
```bash
# training
python train.py --shot $shot --save_path $save_path
# evaluation
python test.py --save_path $save_path --dataset $dataset
# (Optional) we provide bash script for training and evaluating all the datasets
bash scripts.sh
```
Model definition is in ``./model/``. We thank [```open_clip```](https://github.com/mlfoundations/open_clip.git) for being open-source. To run the code, one has to download the weight of OpenCLIP ViT-L-14-336px and put it under ```./model/```.

### 4. Training Results
Training results are saved in the directory specified by ``--save_path`` parameter (default: ``ckpt/baseline``). The saved files include:

- **Checkpoints**:
  - ``text_adapter.pth``: Text adapter checkpoint
  - ``image_adapter.pth``: Latest image adapter checkpoint
  - ``image_adapter_{epoch}.pth``: Image adapter checkpoint for each epoch

- **Training Log**:
  - ``train.log``: Training process log with loss values and other metrics

These files are essential for model evaluation, inference, and resuming training if needed.

### 5. Testing Phase

#### 5.1 Testing Command Format

```bash
python test.py --dataset <dataset_name> --save_path <model_save_path> [other parameters]
```

#### 5.2 Main Parameters

- ``--dataset``: Dataset name (required), supports all configured datasets including the newly added ``MVTec2``
- ``--save_path``: Model save path (required), pointing to the checkpoint directory saved during training
- ``--model_name``: Model name, default is ``ViT-L-14-336``
- ``--img_size``: Image size, default is ``518``
- ``--batch_size``: Batch size, default is ``32``
- ``--visualize``: Whether to visualize test results

#### 5.3 Testing All Datasets

##### 5.3.1 Testing MVTec Dataset
```bash
python test.py --dataset MVTec --save_path ckpt/baseline
```

##### 5.3.2 Testing VisA Dataset
```bash
python test.py --dataset VisA --save_path ckpt/baseline
```

##### 5.3.3 Testing MPDD Dataset
```bash
python test.py --dataset MPDD --save_path ckpt/baseline
```

##### 5.3.4 Testing BTAD Dataset
```bash
python test.py --dataset BTAD --save_path ckpt/baseline
```

##### 5.3.5 Testing Brain Dataset
```bash
python test.py --dataset Brain --save_path ckpt/baseline
```

##### 5.3.6 Testing Liver Dataset
```bash
python test.py --dataset Liver --save_path ckpt/baseline
```

##### 5.3.7 Testing Retina Dataset
```bash
python test.py --dataset Retina --save_path ckpt/baseline
```

##### 5.3.8 Testing Colon-related Datasets
```bash
# CVC-ClinicDB
python test.py --dataset Colon_clinicDB --save_path ckpt/baseline

# CVC-ColonDB
python test.py --dataset Colon_colonDB --save_path ckpt/baseline

# CVC-300
python test.py --dataset Colon_cvc300 --save_path ckpt/baseline

# Kvasir
python test.py --dataset Colon_Kvasir --save_path ckpt/baseline
```

##### 5.3.9 Testing the Newly Added MVTec2 Dataset
```bash
python test.py --dataset MVTec2 --save_path ckpt/baseline
```

#### 5.4 Test Results

After testing, the following files will be generated in the ``--save_path`` directory:

- ``test.log``: Test process log with detailed evaluation metrics
- (Optional) Visualization results: If using the ``--visualize`` parameter, visualization images of test results will be generated

##### 5.4.1 Evaluation Metrics

The test script calculates the following evaluation metrics:

- **pixel AUC**: Pixel-level anomaly detection AUC value
- **pixel AP**: Pixel-level anomaly detection AP value
- **image AUC**: Image-level anomaly detection AUC value
- **image AP**: Image-level anomaly detection AP value

These metrics are calculated per class and an average is generated.

#### 5.5 Notes

1. Ensure that the parameters used during training are consistent with those during testing, especially ``--model_name``, ``--img_size``, etc.
2. Ensure that the ``--save_path`` directory contains the checkpoint files generated during training
3. When testing the MVTec2 dataset, ensure that the dataset is correctly configured and located in the ``/root/autodl-tmp/datasets/mvtec2`` directory
4. For large datasets, it is recommended to use a larger ``--batch_size`` to improve testing speed

#### 5.6 Example: Complete Testing Process

```bash
# 1. Train the model (taking MVTec2 as an example)
python train.py --dataset MVTec2 --shot 0 --save_path ckpt/mvtec2

# 2. Test the model
python test.py --dataset MVTec2 --save_path ckpt/mvtec2 --visualize

# 3. View test results
cat ckpt/mvtec2/test.log
```

#### 5.7 Automatic Testing for All Datasets

We provide two scripts to automatically test all datasets (including the newly added MVTec2) and save the results to a `results` directory.

##### 5.7.1 Python Script (Recommended)

```bash
python test_all_datasets.py --save_path <model_save_path> [--results_dir <results_directory>] [--max_workers <number_of_parallel_tests>]
```

##### 5.7.2 Bash Script

We also provide a bash script `test.sh` that follows the same structure as the original `scripts.sh` file:

```bash
#!/bin/bash

# 创建 results 目录
mkdir -p ./results

# 定义数据集数组
declare -a dataset=(MVTec BTAD MPDD Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir Colon_cvc300 VisA MVTec2)

# 设置 checkpoint 路径
save_path="./ckpt"

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
```

##### 5.7.3 Usage

###### Python Script

```bash
python test_all_datasets.py --save_path <model_save_path> [--results_dir <results_directory>] [--max_workers <number_of_parallel_tests>]
```

###### Bash Script

```bash
chmod +x test.sh
./test.sh
```

##### 5.7.4 Parameters

###### Python Script

- `--save_path`：Model save path (required), pointing to the checkpoint directory saved during training
- `--results_dir`：Directory to save test results, default is `results`
- `--max_workers`：Maximum number of parallel tests, default is `4`

###### Bash Script

- Uses `./ckpt` as the default checkpoint path
- Saves results to `./results` directory

##### 5.7.5 Example

###### Python Script

```bash
python test_all_datasets.py --save_path ckpt/baseline --max_workers 4
```

###### Bash Script

```bash
chmod +x test.sh
./test.sh
```

##### 5.7.6 Results

###### Python Script

After running the script, the following structure will be created in the `results` directory:

```
results/
├── summary.txt          # Test summary for all datasets
├── Brain/
│   └── test.log         # Test results for Brain dataset
├── Liver/
│   └── test.log         # Test results for Liver dataset
├── Retina/
│   └── test.log         # Test results for Retina dataset
├── Colon_clinicDB/
│   └── test.log         # Test results for Colon_clinicDB dataset
├── Colon_colonDB/
│   └── test.log         # Test results for Colon_colonDB dataset
├── Colon_cvc300/
│   └── test.log         # Test results for Colon_cvc300 dataset
├── Colon_Kvasir/
│   └── test.log         # Test results for Colon_Kvasir dataset
├── BTAD/
│   └── test.log         # Test results for BTAD dataset
├── MPDD/
│   └── test.log         # Test results for MPDD dataset
├── MVTec/
│   └── test.log         # Test results for MVTec dataset
├── VisA/
│   └── test.log         # Test results for VisA dataset
└── MVTec2/
    └── test.log         # Test results for MVTec2 dataset
```

###### Bash Script

After running the script, the following structure will be created in the `results` directory:

```
results/
├── MVTec_test.log         # Test results for MVTec dataset
├── BTAD_test.log          # Test results for BTAD dataset
├── MPDD_test.log          # Test results for MPDD dataset
├── Brain_test.log         # Test results for Brain dataset
├── Liver_test.log         # Test results for Liver dataset
├── Retina_test.log        # Test results for Retina dataset
├── Colon_clinicDB_test.log # Test results for Colon_clinicDB dataset
├── Colon_colonDB_test.log  # Test results for Colon_colonDB dataset
├── Colon_Kvasir_test.log   # Test results for Colon_Kvasir dataset
├── Colon_cvc300_test.log   # Test results for Colon_cvc300 dataset
├── VisA_test.log           # Test results for VisA dataset
└── MVTec2_test.log         # Test results for MVTec2 dataset
```

## Additional Discussion
(I am writing down my experimental observations and thoughts. In this part, it is less formal and rigorous.)
We have observed several interesting phenomenons during our experiments:

### The impact of class-level supervision differs across domains.
In the initial stage of adapting the text encoder, we tried to apply binary cross-entropy (BCE) loss to directly distinguish between normal and anomalous text embeddings—an approach that imposes stronger supervision than our current method, which adds class embeddings to visual embeddings and relies on segmentation loss for separation. Experimental results indicate that BCE improves zero-shot performance on industrial datasets but negatively affects performance in the medical domain. This may be due to the lower diversity and simpler structure of anomaly representations in medical data, making them easier to learn without strong supervision.

### Adaptation hyper-parameters should be carefully tuned.
 Since CLIP is pre-trained on a massive dataset and anomaly detection is a comparatively simpler task, the model is prone to issues like catastrophic forgetting or overfitting. Careful control of the adaptation process is essential, which is why our method involves multiple hyper-parameters (though these can still be further optimized).

Additionally, the differences in anomaly patterns between the training dataset and downstream zero-shot datasets cannot be ignored. Overfitting to the anomaly characteristics of the training set can lead the model to rely on superficial cues. For instance, if the training data predominantly features round-shaped anomalies, the model may prioritize shape over true semantic understanding of anomalies. Incorporating training data with a wider variety of anomaly types could help mitigate this issue.

### To be updated...

## Citation
If you use this work, please cite:
```
@misc{ma2025aaclipenhancingzeroshotanomaly,
      title={AA-CLIP: Enhancing Zero-shot Anomaly Detection via Anomaly-Aware CLIP}, 
      author={Wenxin Ma and Xu Zhang and Qingsong Yao and Fenghe Tang and Chenxu Wu and Yingtai Li and Rui Yan and Zihang Jiang and S. Kevin Zhou},
      year={2025},
      eprint={2503.06661},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.06661}, 
}
```

## Contact
For questions or collaborations:

- Email: mwxisj@gmail.com

- GitHub Issues: [Open Issue](https://github.com/Mwxinnn/AA-CLIP/issues)

⭐ Star this repo if you find it useful!