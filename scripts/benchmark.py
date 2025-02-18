import time
import numpy as np
import torch.cuda


class TensorRTBenchmark:
    def __init__(self, trt_engine, pytorch_model):
        self.trt_engine = trt_engine
        self.pytorch_model = pytorch_model
    def run_benchmark(self, dummy_input, num_runs=100):
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