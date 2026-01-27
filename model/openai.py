""" OpenAI pretrained model functions

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import os
import warnings
from typing import List, Optional, Union

import torch

from .model import build_model_from_openai_state_dict, convert_weights_to_lp, get_cast_dtype

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
        print(f"Number of keys: {len(state_dict.keys())}")
        print("First 20 keys:")
        for i, key in enumerate(list(state_dict.keys())[:20]):
            print(f"  {i}: {key}")
        
        # Try different approaches to build the model
        try:
            model = build_model_from_openai_state_dict(state_dict or model.state_dict(), cast_dtype=cast_dtype)
        except KeyError as e:
            print(f"\nError occurred: {e}")
            print("Trying different model formats...")
            
            # Approach 1: Check if it's a Hugging Face format model
            if any(k.startswith("vision.") for k in state_dict.keys()):
                print("Detected Hugging Face format model")
                # Convert Hugging Face format to OpenAI format
                sd = {}
                for k, v in state_dict.items():
                    if k.startswith("vision."):
                        # Remove "vision." prefix to match OpenAI format
                        new_key = k[7:]
                        sd[new_key] = v
                    elif k.startswith("text."):
                        # For text keys, remove "text." prefix
                        new_key = k[5:]
                        sd[new_key] = v
                    elif k in ["logit_scale", "text_projection"]:
                        # These keys are the same in both formats
                        sd[k] = v
                # Try building model with converted state dict
                try:
                    model = build_model_from_openai_state_dict(sd, cast_dtype=cast_dtype)
                    print("Successfully converted Hugging Face format to OpenAI format")
                except Exception as e2:
                    print(f"Failed to convert Hugging Face format: {e2}")
            
            # Approach 2: Handle nested state_dict
            elif "state_dict" in state_dict:
                print("Detected nested state_dict format")
                sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
                try:
                    model = build_model_from_openai_state_dict(sd, cast_dtype=cast_dtype)
                    print("Successfully loaded nested state_dict format")
                except Exception as e2:
                    print(f"Failed to load nested state_dict format: {e2}")
            
            # Approach 3: Check if it's already in OpenAI format but with different structure
            elif any(k.startswith("visual.") for k in state_dict.keys()):
                print("Detected OpenAI format but with different structure")
                try:
                    # Try to build model with existing state_dict
                    model = build_model_from_openai_state_dict(state_dict, cast_dtype=cast_dtype)
                    print("Successfully loaded OpenAI format model")
                except Exception as e2:
                    print(f"Failed to load OpenAI format model: {e2}")
            
            # Approach 4: Try to detect model type based on available keys
            print("Trying to detect model type based on available keys...")
            
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
                if "visual" in k or "conv" in k:
                    print(f"  {k}")
            
            # If all approaches fail, raise detailed error
            error_msg = "\n" + "="*80
            error_msg += "\nERROR: Failed to load model file"
            error_msg += "\n" + "="*80
            error_msg += "\nModel file structure:\n"
            error_msg += f"Number of keys: {len(state_dict.keys())}\n"
            error_msg += "First 20 keys:\n"
            for i, key in enumerate(list(state_dict.keys())[:20]):
                error_msg += f"  {i}: {key}\n"
            error_msg += "\nPossible solutions:\n"
            error_msg += "1. Ensure you're using an OpenAI format CLIP model\n"
            error_msg += "2. Try downloading the model from a different source\n"
            error_msg += "3. Check if the model file is corrupted\n"
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