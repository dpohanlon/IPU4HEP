import sys
import os
import re

from tqdm import tqdm

import json

import subprocess as sp

import numpy as np

batch_sizes = [2 ** x for x in range(16, 24)][::-1]

print(batch_sizes)

timingsGPU = {}
timingsGPUNoSkip = {}

for b in batch_sizes:

    name = f'{b}'

    commandGPU = f"python kfTFVec.py -n {b}"
    commandGPUNoSkip = f"python kfTFVecNoSkip.py -n {b}"

    sp.check_output(commandGPU, shell=True)

    tot = 0
    for i in range(10):

        resultGPU = sp.check_output(commandGPU, shell=True)
        resultGPU = float(str(resultGPU, "utf-8").strip("\n"))
        tot += resultGPU

    timingsGPU[name] = tot/10.

    commandGPU = f"python kfTFVec.py -n {b}"

    sp.check_output(commandGPU, shell=True)

    tot = 0
    for i in range(10):

        resultGPU = sp.check_output(commandGPUNoSkip, shell=True)
        resultGPU = float(str(resultGPU, "utf-8").strip("\n"))
        tot += resultGPU

    timingsGPUNoSkip[name] = tot/10.

json.dump(timingsGPU, open("kf_bench_tf_skip_gpu.json", "w"))
json.dump(timingsGPUNoSkip, open("kf_bench_tf_noSkip_gpu.json", "w"))
