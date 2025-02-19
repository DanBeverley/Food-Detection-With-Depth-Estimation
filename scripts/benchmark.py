import time
import torch.cuda
import numpy as np
from typing import Tuple, List, Union
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTBenchmark:
    def __init__(self, trt_engine, pytorch_model):
        self.trt_engine = trt_engine
        self.pytorch_model = pytorch_model

    def _allocate_buffers(self, batch_size: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], cuda.Stream]:
        """
        Allocate host and device buffers for all bindings in the TensorRT engine.

        Returns:
            inputs: List of host input buffers.
            outputs: List of host output buffers.
            bindings: List of device memory pointers (as ints).
            stream: A CUDA stream for asynchronous execution.
        """
        inputs: List[np.ndarray] = []
        outputs: List[np.ndarray] = []
        bindings: List[int] = []
        stream = cuda.Stream()
        for binding in self.trt_engine:
            binding_shape = self.trt_engine.get_binding_shape(binding)
            size = int(trt.volume(binding_shape)) * batch_size
            dtype = trt.nptype(self.trt_engine.get_binding_dtype(binding))
            # Allocate pagelocked host memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            # Allocate device memory
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.trt_engine.binding_is_input(binding):
                inputs.append(host_mem)
            else:
                outputs.append(host_mem)
        return inputs, outputs, bindings, stream

    def _benchmark_trt(self, dummy_input: Union[torch.Tensor, np.ndarray], num_runs: int = 100) -> float:
        """
        Benchmark the TensorRT engine using the provided dummy input.

        Args:
            dummy_input: A torch.Tensor or numpy array representing a batch of inputs.
            num_runs: Number of runs to average over.

        Returns:
            Average inference time per run in seconds.
        """
        # Convert dummy_input to numpy if needed
        if isinstance(dummy_input, torch.Tensor):
            dummy_input_np = dummy_input.cpu().numpy()
        else:
            dummy_input_np = dummy_input

        batch_size = dummy_input_np.shape[0]
        context = self.trt_engine.create_execution_context()
        inputs, outputs, bindings, stream = self._allocate_buffers(batch_size)

        # Copy dummy input data into the first input buffer (assuming a single input)
        np.copyto(inputs[0], dummy_input_np.ravel())

        # Warm up the TensorRT engine
        for _ in range(10):
            context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
            stream.synchronize()

        start = time.time()
        for _ in range(num_runs):
            context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
            stream.synchronize()
        total_time = time.time() - start
        avg_time = total_time / num_runs
        return avg_time

    def run_benchmark(self, dummy_input, num_runs:int=100) -> dict:
        # Warm up
        for _ in range(10):
            self.pytorch_model(dummy_input)
        # Pytorch timing
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_runs):
            _ = self.pytorch_model(dummy_input)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start)/num_runs

        # TensorRT Timing
        trt_time = self._benchmark_trt(dummy_input, num_runs)
        return {"pytorch_fps":1/pytorch_time,
                "tensorrt_fps":1/trt_time,
                "speedup_factor":(pytorch_time/trt_time)}