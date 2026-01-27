import os
import requests
from tqdm import tqdm  # 自动安装，用于显示下载进度

def download_official_clip_weights(save_path: str):
    """
    从OpenAI官方源下载ViT-L-14-336px.pt权重文件
    
    Args:
        save_path: 完整的保存路径（含文件名）
    """
    # OpenAI官方CLIP权重下载链接（ViT-L/14@336px）
    CLIP_URL = "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f027/ViT-L-14-336px.pt"
    
    # 创建保存目录
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 发送请求（支持断点续传）
        response = requests.get(CLIP_URL, stream=True, timeout=30)
        response.raise_for_status()  # 抛出HTTP错误
        
        # 获取文件总大小
        total_size = int(response.headers.get("content-length", 0))
        # 显示下载进度条
        with open(save_path, "wb") as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        
        # 验证下载结果
        if os.path.getsize(save_path) == total_size:
            print(f"✅ OpenAI官方权重已成功下载到：{save_path}")
        else:
            print("❌ 下载失败：文件大小不匹配，可能是网络中断")
            
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP错误：{e}（可能是官方链接失效）")
    except requests.exceptions.ConnectionError:
        print("❌ 网络连接错误，请检查网络或代理")
    except Exception as e:
        print(f"❌ 下载失败：{str(e)}")

# ===================== 核心配置 =====================
if __name__ == "__main__":
    SAVE_PATH = "/root/autodl-tmp/AA-CLIP_add_mvtec2/model/ViT-L-14-336px.pt"
    download_official_clip_weights(SAVE_PATH)