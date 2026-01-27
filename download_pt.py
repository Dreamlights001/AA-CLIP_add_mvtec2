import os
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError

def download_model_from_hf(
    repo_id: str,          # Hugging Face仓库ID
    filename: str,         # 要下载的文件名
    save_dir: str,         # 保存目录
    force_download: bool = False  # 是否强制重新下载（覆盖已存在文件）
):
    """
    从Hugging Face下载单个模型文件到指定目录
    
    Args:
        repo_id: Hugging Face仓库ID，例如 "openai/clip-vit-large-patch14-336"
        filename: 要下载的文件名，例如 "ViT-L-14-336px.pt"
        save_dir: 本地保存目录
        force_download: 是否强制重新下载（即使文件已存在）
    """
    # 1. 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)
    print(f"保存目录已确认/创建：{save_dir}")
    
    try:
        # 2. 下载文件（自动处理断点续传）
        downloaded_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=save_dir,  # 直接保存到目标目录（而非默认缓存目录）
            force_download=force_download,
            resume_download=True  # 支持断点续传
        )
        
        # 3. 验证下载结果
        if os.path.exists(downloaded_file_path):
            print(f"✅ 模型下载成功！文件路径：{downloaded_file_path}")
            return downloaded_file_path
        else:
            print("❌ 模型下载失败：文件未找到")
            return None
    
    except RepositoryNotFoundError:
        print(f"❌ 错误：Hugging Face仓库 {repo_id} 不存在，请检查repo_id是否正确")
    except HFValidationError:
        print(f"❌ 错误：文件 {filename} 在仓库 {repo_id} 中不存在，请核对文件名")
    except PermissionError:
        print(f"❌ 错误：没有权限写入目录 {save_dir}，请检查目录权限")
    except Exception as e:
        print(f"❌ 下载过程中发生未知错误：{str(e)}")
        return None

# ===================== 核心配置（根据你的需求修改） =====================
if __name__ == "__main__":
    # 配置参数
    REPO_ID = "openai/clip-vit-large-patch14-336"  # ViT-L-14-336px.pt 对应的Hugging Face仓库
    FILENAME = "ViT-L-14-336px.pt"                 # 要下载的模型文件名
    SAVE_DIR = "/root/autodl-tmp/AA-CLIP_add_mvtec2/model/"  # 目标保存目录
    
    # 执行下载
    download_model_from_hf(
        repo_id=REPO_ID,
        filename=FILENAME,
        save_dir=SAVE_DIR,
        force_download=False  # 如果文件已存在，跳过下载；需要重新下载则设为True
    )