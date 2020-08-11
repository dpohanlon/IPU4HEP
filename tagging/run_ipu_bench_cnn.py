import sys
import os
import re

from tqdm import tqdm

import json

import subprocess as sp

import numpy as np

batch_sizes = [256, 128, 32]
input_lengths = [256, 128, 32]
input_features = [32, 16, 4]
hidden_size = [128, 32, 16]
conv_k1 = [2, 10, 25, 50]
conv_k2 = [2, 10, 25, 50]

timingsCPU = {}
timingsIPU = {}
timingsGPU = {}

cpu = False
ipu = True
gpu = False

for b in tqdm(batch_sizes):
    for l in input_lengths:
        for f in input_features:
            for h in hidden_size:
                for k1 in conv_k1:
                    for k2 in conv_k2:

                        name = f'{b}-{l}-{f}-{h}-{k1}-{k2}'

                        if ipu:

                            commandIPU = f"python cnn_model_bench.py -n {b} -t {l} -s {h} -f {f} --k1 {k1} --k2 {k2}"
                            try:
                                resultIPU = sp.check_output(commandIPU, shell=True)
                                resultIPU = float(str(resultIPU, "utf-8").strip("\n"))
                            except:
                                resultIPU = -1

                            timingsIPU[name] = resultIPU

                        if cpu:
                            commandCPU = f"python cnn_model_bench.py -n {b} -t {l} -s {h} -f {f}  --k1 {k1} --k2 {k2} -c"

                            try:
                                resultCPU = sp.check_output(commandCPU, shell=True)
                                resultCPU = float(str(resultCPU, "utf-8").strip("\n"))
                            except:
                                resultCPU = -1

                            timingsCPU[name] = resultCPU

                        if gpu:
                            commandGPU = f"python cnn_model_bench.py -n {b} -t {l} -s {h} -f {f}  --k1 {k1} --k2 {k2} -g"

                            try:
                                resultGPU = sp.check_output(commandGPU, shell=True)
                                resultGPU = float(str(resultGPU, "utf-8").strip("\n"))
                            except:
                                resultGPU = -1

                            timingsGPU[name] = resultGPU

if cpu: json.dump(timingsCPU, open("param_bench_cnn_cpu.json", "w"))
if ipu: json.dump(timingsIPU, open("param_bench_cnn_ipu.json", "w"))
if gpu: json.dump(timingsGPU, open("param_bench_cnn_gpu.json", "w"))
