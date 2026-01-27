import os
import torch
from huggingface_hub import snapshot_download
from transformers import CLIPModel

def download_and_convert_clip_model(
    repo_id: str,
    save_dir: str,
    target_filename: str = "ViT-L-14-336px.pt"
):
    """
    从Hugging Face仓库下载CLIP模型权重，并转换为指定的.pt文件
    
    Args:
        repo_id: Hugging Face仓库ID（openai/clip-vit-large-patch14-336）
        save_dir: 最终文件保存目录
        target_filename: 原代码需要的目标文件名
    """
    # 1. 定义路径
    final_save_path = os.path.join(save_dir, target_filename)
    # 临时缓存目录（下载仓库权重用）
    cache_dir = os.path.join(save_dir, "hf_cache")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    try:
        # 2. 从Hugging Face仓库下载所有权重文件（断点续传）
        print(f"📥 开始从 {repo_id} 下载模型权重...")
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            resume_download=True,  # 断点续传
            local_dir_use_symlinks=False  # 避免符号链接问题
        )
        print("✅ 仓库权重下载完成")

        # 3. 加载模型（从缓存目录加载，避免重复下载）
        print("🔧 加载模型并转换格式...")
        model = CLIPModel.from_pretrained(
            repo_id,
            cache_dir=cache_dir,
            trust_remote_code=True,  # 兼容新版transformers
            local_files_only=True  # 仅使用本地下载的权重，不联网
        )

        # 4. 保存为原代码需要的.pt文件（两种方式可选，根据原代码适配）
        # 方式1：保存完整模型（含结构，原代码可直接torch.load加载）
        torch.save(model, final_save_path)
        # 方式2：仅保存权重（若原代码有模型结构定义，文件更小，注释掉方式1启用此方式）
        # torch.save(model.state_dict(), final_save_path)

        # 5. 验证结果
        if os.path.exists(final_save_path) and os.path.getsize(final_save_path) > 0:
            print(f"✅ 模型转换完成！最终文件路径：{final_save_path}")
            print(f"📌 文件大小：{os.path.getsize(final_save_path)/1024/1024:.1f}MB")
            
            # 可选：删除临时缓存（节省空间）
            # import shutil
            # shutil.rmtree(cache_dir)
        else:
            print("❌ 模型保存失败：文件为空或未生成")

    except Exception as e:
        print(f"❌ 执行失败：{str(e)}")
        print("\n💡 排查建议：")
        print("  1. 确认已执行 export HF_ENDPOINT=https://hf-mirror.com")
        print("  2. 检查网络是否能访问 https://hf-mirror.com")
        print("  3. 确保磁盘空间足够（至少2GB）")

# ===================== 执行配置 =====================
if __name__ == "__main__":
    # Hugging Face仓库ID（固定为openai/clip-vit-large-patch14-336）
    REPO_ID = "openai/clip-vit-large-patch14-336"
    # 目标保存目录（匹配原代码报错路径）
    SAVE_DIR = "/root/autodl-tmp/AA-CLIP_add_mvtec2/model/"
    # 目标文件名（原代码需要的ViT-L-14-336px.pt）
    TARGET_FILENAME = "ViT-L-14-336px.pt"

    # 执行下载+转换
    download_and_convert_clip_model(
        repo_id=REPO_ID,
        save_dir=SAVE_DIR,
        target_filename=TARGET_FILENAME
    )