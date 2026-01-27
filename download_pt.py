import os
import torch
from transformers import CLIPVisionModel, CLIPProcessor, CLIPModel

def save_clip_model_to_pt(save_path: str):
    """
    加载CLIP ViT-L-14-336模型并保存为.pt文件
    
    Args:
        save_path: 完整的保存路径（含文件名），例如 "/root/autodl-tmp/AA-CLIP_add_mvtec2/model/ViT-L-14-336px.pt"
    """
    # 1. 创建保存目录（如果不存在）
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    print(f"保存目录已确认/创建：{save_dir}")
    
    try:
        # 2. 加载OpenAI的CLIP ViT-L-14-336模型（完整模型，含视觉和文本分支）
        # 若原代码仅需视觉分支，可替换为 CLIPVisionModel.from_pretrained(...)
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        
        # 3. 保存模型权重（包含模型结构+权重，或仅权重，根据原代码需求选择）
        # 方式1：保存完整模型（含结构，原代码可直接torch.load加载）
        torch.save(model, save_path)
        # 方式2：仅保存权重（若原代码有模型结构定义，推荐此方式，文件更小）
        # torch.save(model.state_dict(), save_path)
        
        # 4. 验证保存结果
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            print(f"✅ CLIP模型已成功保存到：{save_path}")
            # 验证加载（可选）
            test_load = torch.load(save_path, map_location="cpu")
            print(f"✅ 模型文件可正常加载，模型类型：{type(test_load)}")
        else:
            print("❌ 模型保存失败：文件为空或未生成")
            
    except Exception as e:
        print(f"❌ 模型加载/保存失败：{str(e)}")

# ===================== 核心配置 =====================
if __name__ == "__main__":
    # 目标保存路径（匹配原代码报错的路径）
    SAVE_PATH = "/root/autodl-tmp/AA-CLIP_add_mvtec2/model/ViT-L-14-336px.pt"
    save_clip_model_to_pt(SAVE_PATH)