import logging
from typing import Tuple, Optional, Dict

import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic
import tensorrt as trt


class ModelOptimizer:
    @staticmethod
    def quantize_model(model:nn.Module, qconfig_spec:Tuple[type, ...]=(torch.nn.Linear, torch.nn.Conv2d),
                       dtype: torch.dtype = torch.qint8):
        """
        Apply dynamic quantization to the model for CPU deployment.

        Args:
            model (nn.Module): The model to quantize.
            qconfig_spec (Tuple[type, ...]): The layer types to quantize (default: nn.Linear, nn.Conv2d).
            dtype (torch.dtype): The data type for quantization (default: torch.qint8).

         Returns:
            nn.Module: The quantized model.
        """
        if next(model.parameters()).device.type != "cpu":
            raise RuntimeError("Model must be on CPU for dynamic quantization")
        return quantize_dynamic(model.to('cpu'), qconfig_spec, dtype=dtype)

    @staticmethod
    def export_onnx(model:nn.Module, input_shape:Tuple[int,...],
                    onnx_path:str="model.onnx", dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None) -> None:
        """
        Export the model to ONNX format.

        Args:
            model (nn.Module): The model to export.
            input_shape (Tuple[int, ...]): The shape of the input tensor (excluding batch dimension).
            onnx_path (str): The path to save the ONNX model.
            dynamic_axes (Optional[Dict[str, Dict[int, str]]]): Dynamic axes for ONNX export (default: None).
        """
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, *input_shape, device=device)
        if dynamic_axes is None:
            dynamic_axes = {"input":{0:"batch_size"},
                            "output":{0:"batch_size"}}
        try:
            torch.onnx.export(model, (dummy_input,), onnx_path, dynamic_axes=dynamic_axes,
                              input_names=["input"], output_names=["output"])
        except Exception as e:
            logging.error(f"ONNX export failed: {e}")
            raise

    @staticmethod
    def export_tensorrt(onnx_path:str, input_shape:Tuple[int, ...]=(3, 640, 640),precision:str="FP32",
                        output_path:str="model.trt", max_batch_size:int=1) -> trt.ICudaEngine:
        """
        Export an ONNX model to a TensorRT engine.

        Args:
            onnx_path (str): Path to the ONNX model.
            input_shape (Tuple[int, ...]): Shape of the input tensor (including batch dimension).
            output_path (str): Path to save the TensorRT engine.

        Returns:
            trt.ICudaEngine: The built TensorRT engine.
        """
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = 1<<30 # 1 GB
        if precision == "FP16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "INT8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Parse ONNX model
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, "rb") as model_file:
            if not parser.parse(model_file.read()):
                logging.error("Failed to parse ONNX model")
                raise RuntimeError("ONNX parsing failed")
        builder.max_batch_size = max_batch_size
        # Build engine
        try:
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            with open(output_path, "wb") as f:
                f.write(engine.serialize())
            return engine
        except Exception as e:
            logging.error(f"TensorRT engine build failed: {e}")
            raise
