from typing import Tuple

import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic
import tensorrt as trt


class ModelOptimizer:
    @staticmethod
    def quantize_model(model:nn.Module, qconfig_spec:Tuple[type, ...]=(torch.nn.Linear, torch.nn.Conv2d)):
        """Dynamic quantization for CPU deployment"""
        return quantize_dynamic(
            model.to('cpu'),
            qconfig_spec,
            dtype=torch.qint8
        )

    @staticmethod
    def export_onnx(model:nn.Module, input_shape:Tuple[int,...], onnx_path:str="model.onnx") -> None:
        dummy_input = torch.randn(1, *input_shape)
        torch.onnx.export(model, dummy_input, onnx_path)

    @staticmethod
    def export_tensorrt(model:nn.Module, input_shape:Tuple[int, ...]=(3, 640, 640),
                        output_path:str="model.trt") -> trt.ICudaEngine:
        """Export model to TensorRT engine"""
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = 1<<30 # 1 GB
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Define input configuration
        input_tensor = network.add_input(name="input", dtype=trt.float32, shape=input_shape)

        # Convert PyTorch model to ONNX then TensorRT
        # (Implementation details would depend on specific model architecture)

        # Serialize and save engine
        engine = builder.build_engine(network, config)
        with open(output_path, "wb") as f:
            f.write(engine.serialize())

        return engine