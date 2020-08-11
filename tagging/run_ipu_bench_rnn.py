import sys
import os
import re

from tqdm import tqdm

import json

import subprocess as sp

import numpy as np

batch_sizes = [2 ** x for x in range(5, 10)][::-1]
input_lengths = [2 ** x for x in range(5, 10)][::-1]
input_features = [2 ** x for x in range(0, 8)][::-1]
hidden_size = [2 ** x for x in range(2, 8)][::-1]

timingsCPU = {}
timingsIPU = {}
timingsGPU = {}

cpu = True
ipu = False
gpu = False

for b in batch_sizes:
    for l in tqdm(input_lengths):
        for f in tqdm(input_features):
            for h in hidden_size:

                name = f'{b}-{l}-{f}-{h}'

                if ipu:

                    commandIPU = f"python rnn_model_bench.py -n {b} -t {l} -s {h} -f {f}"
                    try:
                        resultIPU = sp.check_output(commandIPU, shell=True)
                        resultIPU = float(str(resultIPU, "utf-8").strip("\n"))
                    except:
                        resultIPU = -1

                    timingsIPU[name] = resultIPU

                if cpu:

                    commandCPU = f"python rnn_model_bench.py -n {b} -t {l} -s {h} -f {f} -c"
                    resultCPU = sp.check_output(commandCPU, shell=True)
                    resultCPU = float(str(resultCPU, "utf-8").strip("\n"))

                    timingsCPU[name] = resultCPU

                if gpu:

                    commandGPU = f"python rnn_model_bench.py -n {b} -t {l} -s {h} -f {f} -g"
                    resultGPU = sp.check_output(commandGPU, shell=True)
                    resultGPU = float(str(resultGPU, "utf-8").strip("\n"))

                    timingsGPU[name] = resultGPU

if cpu: json.dump(timingsCPU, open("model_size_bench_rnn_cpu.json", "w"))
if ipu: json.dump(timingsIPU, open("param_bench_rnn_ipu.json", "w"))
if gpu: json.dump(timingsGPU, open("param_bench_rnn_gpu.json", "w"))
