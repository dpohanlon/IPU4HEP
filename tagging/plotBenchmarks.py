import matplotlib.pyplot as plt

from matplotlib import rcParams
import matplotlib as mpl
mpl.use('Agg')

plt.style.use(['seaborn-whitegrid', 'seaborn-ticks'])
import matplotlib.ticker as plticker
rcParams['figure.figsize'] = 12, 8
rcParams['axes.facecolor'] = 'FFFFFF'
rcParams['savefig.facecolor'] = 'FFFFFF'
rcParams['figure.facecolor'] = 'FFFFFF'

rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

rcParams['mathtext.fontset'] = 'cm'
rcParams['mathtext.rm'] = 'serif'

rcParams.update({'figure.autolayout': True})

import numpy as np
from tqdm import tqdm
import json
import pandas as pd
import re

import seaborn as sns

from scipy.stats import binned_statistic

rnn_cpu_file_name = 'param_bench_rnn_cpu.json'
rnn_ipu_file_name = 'param_bench_rnn_ipu.json'
rnn_gpu_file_name = 'param_bench_rnn_gpu.json'

cnn_cpu_file_name = 'param_bench_cnn_cpu.json'
cnn_ipu_file_name = 'param_bench_cnn_ipu.json'
cnn_gpu_file_name = 'param_bench_cnn_gpu.json'

rnnVars = [
'batch_size',
'input_length',
'input_features',
'hidden_size']

cnnVars = [
'batch_size',
'input_length',
'input_features',
'hidden_size',
'k1',
'k2'
]

varNames = {
'batch_size' : 'Batch size',
'input_length' : 'Input length',
'input_features' : 'Input features',
'hidden_size' : 'Hidden size',
'k1' : r'$k_1$',
'k2' : r'$k_2$',
}

def parseRNNBenchName(n):

    matches = re.findall('[0-9]+', n)

    batch_size = int(matches[0])
    input_length = int(matches[1])
    input_features = int(matches[2])
    hidden_size = int(matches[3])

    return [batch_size, input_length, input_features, hidden_size]

def parseCNNBenchName(n):

    matches = re.findall('[0-9]+', n)

    batch_size = int(matches[0])
    input_length = int(matches[1])
    input_features = int(matches[2])
    hidden_size = int(matches[3])
    k1 = int(matches[4])
    k2 = int(matches[5])

    return [batch_size, input_length, input_features, hidden_size, k1, k2]

def makeDF(fileName):

    # modelSizes = json.load(open('model_size_bench_rnn_cpu.json', 'r'))

    rnn = 'rnn' in fileName
    vars = rnnVars if rnn else cnnVars

    data = json.load(open(fileName, 'r'))

    dataD = {v : [] for v in vars}
    dataD['time'] = []
    # dataD['model_size'] = []

    for k, v in data.items():

        d = parseRNNBenchName(k) if rnn else parseCNNBenchName(k)

        dataD['time'].append(float(v) if float(v) > 0 else np.nan)
        # dataD['model_size'].append(int(modelSizes[k]))

        for iv, v in enumerate(vars):
            dataD[v].append(d[iv])

    df = pd.DataFrame(dataD)

    df['time_input'] = df['time'] / df['batch_size']

    df.to_hdf(fileName[:-5] + '.h5', fileName[:-5])

    return df

if __name__ == '__main__':

    # fileNames =  [rnn_cpu_file_name, rnn_ipu_file_name, rnn_gpu_file_name]
    fileNames =  [cnn_cpu_file_name, cnn_ipu_file_name, cnn_gpu_file_name]
    dfNames = ['cpu', 'ipu', 'gpu']

    dfs = list(map(makeDF, fileNames))
    dfs = {dfNames[i] : dfs[i].rename(columns = {'time' : 'time_' + dfNames[i],
                                                 'time_input' : 'time_input_' +dfNames[i]})
                                                 for i in range(len(dfNames))}

    mergedDF = dfs['ipu'].merge(dfs['gpu']).merge(dfs['cpu'])

    mergedDF['gpu/ipu'] =  mergedDF['time_gpu'] / mergedDF['time_ipu'] # Larger -> IPU faster
    mergedDF['cpu/ipu'] =  mergedDF['time_cpu'] / mergedDF['time_ipu']
    mergedDF['cpu/gpu'] =  mergedDF['time_cpu'] / mergedDF['time_gpu']

    mergedDF['gpu/ipu_input'] =  mergedDF['time_input_gpu'] / mergedDF['time_input_ipu'] # Larger -> IPU faster
    mergedDF['cpu/ipu_input'] =  mergedDF['time_input_cpu'] / mergedDF['time_input_ipu']
    mergedDF['cpu/gpu_input'] =  mergedDF['time_input_cpu'] / mergedDF['time_input_gpu']

    print(mergedDF.sort_values(by = 'gpu/ipu', ascending = True)[:25])

    for var in cnnVars:
    # for var in rnnVars:

        grouped = mergedDF.groupby(by = var)

        fig = plt.figure(1)
        ax = fig.add_subplot(111)

        plt.fill_between(grouped.min().index, grouped.min()['gpu/ipu'], grouped.max()['gpu/ipu'], alpha = 0.25)
        plt.plot(grouped.mean().index, grouped.mean()['gpu/ipu'], color = 'k')
        plt.ylabel('Time per input GPU / IPU', fontsize = 28)
        plt.xlabel(varNames[var], fontsize = 28)
        ax.tick_params(axis='both', which='major', labelsize=26)
        plt.savefig(f'{var}-CNN.pdf')
        # plt.savefig(f'{var}-RNN.pdf')
        plt.clf()

    # modelDF = mergedDF.query('model_size < 100000')
    #
    # sizes = modelDF['model_size'][np.isfinite(modelDF['gpu/ipu_input'])].values
    # times = modelDF['gpu/ipu_input'][np.isfinite(modelDF['gpu/ipu_input'])].values
    #
    # h, b, _ = binned_statistic(sizes, times, statistic = 'mean', bins = 25)
    # binCentres = b[:-1] + (b[1] - b[0]) / 2.
    # plt.plot(binCentres, h, '.')
    # plt.savefig('test.pdf')
