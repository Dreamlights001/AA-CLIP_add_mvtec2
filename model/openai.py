""" OpenAI pretrained model functions

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import os
import warnings
from typing import List, Optional, Union

import torch

from .model import build_model_from_openai_state_dict, convert_weights_to_lp, get_cast_dtype

# Try to import OpenCLIP library
try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False
    warnings.warn("OpenCLIP library not available. Install it with 'pip install open_clip_torch'")

__all__ = ["list_openai_models", "load_openai_model"]


def load_openai_model(
        name: str,
        precision: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        jit: bool = True,
        cache_dir: Optional[str] = None,
):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    precision: str
        Model precision, if None defaults to 'fp32' if device == 'cpu' else 'fp16'.
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.
    cache_dir : Optional[str]
        The directory to cache the downloaded model weights

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision is None:
        precision = 'fp32' if device == 'cpu' else 'fp16'

    if os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        # Build a non-jit model from the OpenAI jitted model state dict
        cast_dtype = get_cast_dtype(precision)
        # Diagnostic: Print model file structure
        print("Model file structure:")
        # Check if state_dict is None (JIT model case)
        if state_dict is None:
            if hasattr(model, 'state_dict'):
                state_dict = model.state_dict()
                print(f"Number of keys: {len(state_dict.keys())}")
                print("First 20 keys:")
                for i, key in enumerate(list(state_dict.keys())[:20]):
                    print(f"  {i}: {key}")
            else:
                print("State dict is None (JIT model)")
        else:
            print(f"Number of keys: {len(state_dict.keys())}")
            print("First 20 keys:")
            for i, key in enumerate(list(state_dict.keys())[:20]):
                print(f"  {i}: {key}")
        
        # Try different approaches to build the model
        try:
            model = build_model_from_openai_state_dict(state_dict or model.state_dict(), cast_dtype=cast_dtype)
            print("Successfully loaded model in OpenAI format")
        except KeyError as e:
            print(f"\nError occurred: {e}")
            print("Trying different model formats...")
            
            # Approach 1: Try Hugging Face format conversion first
            print("\nApproach 1: Trying Hugging Face format conversion...")
            try:
                # Check if state_dict is available
                if state_dict is None:
                    print("State dict is None, skipping Hugging Face format conversion...")
                else:
                    print("Converting Hugging Face format to OpenAI format...")
                    # Convert Hugging Face format to OpenAI format
                    sd = {}
                    for k, v in state_dict.items():
                        # Handle text part conversion
                        if k.startswith("text_model."):
                            # Convert text_model.xxx to OpenAI format
                            if k == "text_model.embeddings.token_embedding.weight":
                                sd["token_embedding.weight"] = v
                            elif k == "text_model.embeddings.position_embedding.weight":
                                sd["positional_embedding"] = v
                            elif k == "text_model.final_layer_norm.weight":
                                sd["ln_final.weight"] = v
                            elif k == "text_model.final_layer_norm.bias":
                                sd["ln_final.bias"] = v
                            elif k == "text_model.text_projection.weight":
                                sd["text_projection"] = v
                            elif "text_model.encoder.layers." in k:
                                # Convert transformer layers
                                layer_parts = k.split(".")
                                if len(layer_parts) >= 5:
                                    layer_idx = layer_parts[3]
                                    layer_type = layer_parts[4]
                                    
                                    if layer_type == "self_attn":
                                        # Convert attention layers
                                        if "k_proj.weight" in k:
                                            sd[f"transformer.resblocks.{layer_idx}.attn.in_proj_weight"] = v
                                        elif "v_proj.weight" in k:
                                            # Skip, we'll handle this differently
                                            pass
                                        elif "q_proj.weight" in k:
                                            # Skip, we'll handle this differently
                                            pass
                                        elif "out_proj.weight" in k:
                                            sd[f"transformer.resblocks.{layer_idx}.attn.out_proj.weight"] = v
                                        elif "k_proj.bias" in k:
                                            sd[f"transformer.resblocks.{layer_idx}.attn.in_proj_bias"] = v
                                        elif "v_proj.bias" in k:
                                            # Skip
                                            pass
                                        elif "q_proj.bias" in k:
                                            # Skip
                                            pass
                                        elif "out_proj.bias" in k:
                                            sd[f"transformer.resblocks.{layer_idx}.attn.out_proj_bias"] = v
                                    elif layer_type == "layer_norm1":
                                        if "weight" in k:
                                            sd[f"transformer.resblocks.{layer_idx}.ln_1.weight"] = v
                                        elif "bias" in k:
                                            sd[f"transformer.resblocks.{layer_idx}.ln_1.bias"] = v
                                    elif layer_type == "mlp":
                                        if "fc1.weight" in k:
                                            sd[f"transformer.resblocks.{layer_idx}.mlp.c_fc.weight"] = v
                                        elif "fc1.bias" in k:
                                            sd[f"transformer.resblocks.{layer_idx}.mlp.c_fc.bias"] = v
                                        elif "fc2.weight" in k:
                                            sd[f"transformer.resblocks.{layer_idx}.mlp.c_proj.weight"] = v
                                        elif "fc2.bias" in k:
                                            sd[f"transformer.resblocks.{layer_idx}.mlp.c_proj.bias"] = v
                                    elif layer_type == "layer_norm2":
                                        if "weight" in k:
                                            sd[f"transformer.resblocks.{layer_idx}.ln_2.weight"] = v
                                        elif "bias" in k:
                                            sd[f"transformer.resblocks.{layer_idx}.ln_2.bias"] = v
                        # Handle visual part conversion
                        elif k == "visual_projection.weight":
                            sd["visual.proj"] = v
                        # Handle other keys
                        elif k == "logit_scale":
                            sd[k] = v
                    
                    # Print conversion details for debugging
                    print(f"Converted {len(sd)} keys from Hugging Face format to OpenAI format")
                    print("First 10 converted keys:")
                    for i, key in enumerate(list(sd.keys())[:10]):
                        print(f"  {i}: {key}")
                    
                    # Try building model with converted state dict
                    model = build_model_from_openai_state_dict(sd, cast_dtype=cast_dtype)
                    print("Successfully converted Hugging Face format to OpenAI format")
            except Exception as e2:
                print(f"Failed to convert Hugging Face format: {e2}")
            
            # Approach 2: Try nested state_dict format
            print("\nApproach 2: Trying nested state_dict format...")
            try:
                # Check if state_dict is available
                if state_dict is None:
                    print("State dict is None, skipping nested state_dict format...")
                elif "state_dict" in state_dict:
                    print("Detected nested state_dict format")
                    sd = {}
                    for k, v in state_dict["state_dict"].items():
                        if k.startswith("module."):
                            sd[k[7:]] = v
                        else:
                            sd[k] = v
                    
                    # Print conversion details for debugging
                    print(f"Extracted {len(sd)} keys from nested state_dict")
                    print("First 10 extracted keys:")
                    for i, key in enumerate(list(sd.keys())[:10]):
                        print(f"  {i}: {key}")
                    
                    model = build_model_from_openai_state_dict(sd, cast_dtype=cast_dtype)
                    print("Successfully loaded nested state_dict format")
                else:
                    print("No nested state_dict found")
            except Exception as e2:
                print(f"Failed to load nested state_dict format: {e2}")
            
            # Approach 3: Try to detect model type based on available keys
            print("\nApproach 3: Trying model type detection...")
            try:
                # Check if state_dict is available
                if state_dict is None:
                    print("State dict is None, skipping model type detection...")
                else:
                    # Check for ViT-specific keys
                    if any(k.endswith(".conv1.weight") for k in state_dict.keys()):
                        print("Detected ViT model structure")
                    elif any(k.endswith(".layer1.0.conv1.weight") for k in state_dict.keys()):
                        print("Detected ResNet model structure")
                    else:
                        print("Could not detect model structure")
                    
                    # Print all visual-related keys for debugging
                    print("Visual-related keys:")
                    for k in state_dict.keys():
                        if "visual" in k or "conv" in k or "projection" in k:
                            print(f"  {k}")
                    
                    # Try to build model with existing state_dict again
                    model = build_model_from_openai_state_dict(state_dict, cast_dtype=cast_dtype)
                    print("Successfully loaded model with existing state_dict")
            except Exception as e3:
                print(f"Failed to load model with existing state_dict: {e3}")
            
            # Approach 4: Try using OpenCLIP library
            print("\nApproach 4: Trying OpenCLIP library...")
            try:
                if OPENCLIP_AVAILABLE:
                    # Check if state_dict is available
                    if state_dict is None:
                        print("State dict is None, skipping OpenCLIP library...")
                    else:
                        print("Using OpenCLIP library to load model...")
                        # Try to load the model using OpenCLIP
                        # First, save the state_dict to a temporary file
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                            torch.save(state_dict, tmp.name)
                            tmp_path = tmp.name
                        
                        # Try loading with OpenCLIP
                        model, _, preprocess = open_clip.create_model_and_transforms(
                            "ViT-L-14-336",
                            pretrained=tmp_path,
                            precision=precision,
                            device=device
                        )
                        
                        # Clean up temporary file
                        os.unlink(tmp_path)
                        
                        print("Successfully loaded model using OpenCLIP library")
                else:
                    print("OpenCLIP library not available, skipping...")
            except Exception as e4:
                print(f"Failed to load model using OpenCLIP library: {e4}")
            
            # If all approaches fail, raise detailed error
            error_msg = "\n" + "="*80
            error_msg += "\nERROR: Failed to load model file"
            error_msg += "\n" + "="*80
            error_msg += "\nModel file structure:\n"
            if state_dict is None:
                error_msg += "State dict is None (JIT model)\n"
            else:
                error_msg += f"Number of keys: {len(state_dict.keys())}\n"
                error_msg += "First 20 keys:\n"
                for i, key in enumerate(list(state_dict.keys())[:20]):
                    error_msg += f"  {i}: {key}\n"
            error_msg += "\nPossible solutions:\n"
            error_msg += "1. Ensure you're using an OpenAI format CLIP model\n"
            error_msg += "2. Run 'python download_model.py' to download the correct model\n"
            error_msg += "3. Install OpenCLIP library with 'pip install open_clip_torch'\n"
            error_msg += "4. Check if the model file is corrupted\n"
            error_msg += "\n" + "="*80
            raise RuntimeError(error_msg)

        # model from OpenAI state dict is in manually cast fp16 mode, must be converted for AMP/fp32/bf16 use
        model = model.to(device)
        if precision.startswith('amp') or precision == 'fp32':
            model.float()
        elif precision == 'bf16':
            convert_weights_to_lp(model, dtype=torch.bfloat16)

        return model

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 (typically for CPU)
    if precision == 'fp32':
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()

    # ensure image_size attr available at consistent location for both jit and non-jit
    model.visual.image_size = model.input_resolution.item()
    return model