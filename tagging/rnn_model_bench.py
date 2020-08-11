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

import argparse

import numpy as np

from tqdm import tqdm

from numpy.random import seed
seed(42)

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import torch
torch.set_num_threads(1)

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

try:
    import popart
except:
    pass

import time

cuda = torch.cuda.is_available()
device = "cuda:0" if cuda else "cpu"

class RNNNet(torch.nn.Module):
    def __init__(self, nHidden, nFeatures):
        super(RNNNet, self).__init__()

        hidden_size = nHidden
        input_size = nFeatures

        self.relu = torch.nn.functional.relu

        self.lstm = torch.nn.LSTM(input_size = input_size,
                                  hidden_size = hidden_size,
                                  batch_first = True)
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.linearOut = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, init_hc):

        x, (h, c) = self.lstm(x, (init_hc, init_hc))

        x = h.squeeze()

        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu(x)

        return self.linearOut(x)

    def eval(self, x, init_hc):
        return torch.sigmoid(self.forward(x, init_hc))

if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-n", type = int, dest = "n", default = None, help = 'Batch size.')
    argParser.add_argument("-t", type = int, dest = "nTracks", default = 100, help = 'Number of tracks.')
    argParser.add_argument("-f", type = int, dest = "nFeatures", default = 18, help = 'Number of features.')
    argParser.add_argument("-s", type = int, dest = "nHidden", default = 16, help = 'Hidden size of LSTM.')
    argParser.add_argument("-d", type = int, dest = "s", default = 1, help = 'Number of stacked LSTMs.')
    argParser.add_argument("-c", default = False, action = "store_true", dest = 'cpu', help = 'Run on CPU.')
    argParser.add_argument("-g", default = False, action = "store_true", dest = 'gpu', help = 'Run on GPU.')
    args = argParser.parse_args()

    cpu = args.cpu # Make sure this runs on only one core, set OMP/MKL_NUM_THREADS=1
    gpu = args.gpu
    n = args.n

    nTracks = args.nTracks
    nFeatures = args.nFeatures
    nHidden = args.nHidden

    try:
        net = RNNNet(nHidden, nFeatures)
    except:
        print('-1')

    nParams = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print(nParams)

    dummy_input = torch.randn((n, nTracks, nFeatures), dtype = torch.float)
    dummy_init_hc = torch.torch.zeros([1, n, nHidden])

    if cuda and gpu:
        net = net.to(device)
        dummy_input = dummy_input.to(device)
        dummy_init_hc = dummy_init_hc.to(device)

    export_name = "lstm_workaround.onnx"

    # torch.onnx.export(net, (dummy_input, dummy_init_hc), export_name,
    #                   verbose = False, input_names = ['data', 'init_hc'],
    #                   output_names = ['tag'])

    if cpu or gpu:

        start = time.perf_counter()

        net(dummy_input, dummy_init_hc)

        end = time.perf_counter()

        elapsed = end - start
        timePerEvent = elapsed / n

        print("{:.12f}".format(elapsed))

    else:

        # IMPORT INTO POPART

        graph_transformer = popart.GraphTransformer(export_name)

        inputShapeInfo = popart.InputShapeInfo()
        inputShapeInfo.add("data", popart.TensorInfo("FLOAT", [n, nTracks, nFeatures]))
        inputShapeInfo.add("init_hc", popart.TensorInfo("FLOAT", [1,n,nHidden]))

        anchors = {"tag" : popart.AnchorReturnType("ALL")}
        dataFeed = popart.DataFlow(1, anchors)
        # device = popart.DeviceManager().createIpuModelDevice({})
        device = popart.DeviceManager().acquireAvailableDevice(1)

        session = popart.InferenceSession(graph_transformer.getModelProto(), dataFeed, device, inputShapeInfo=inputShapeInfo)

        session.prepareDevice()

        inferenceAnchors = session.initAnchorArrays()

        data_input = np.random.rand(n, nTracks, nFeatures).astype(np.float32)
        init_hc = np.zeros([1,n,nHidden]).astype(np.float32)

        stepio = popart.PyStepIO({"data": data_input, "init_hc": init_hc}, inferenceAnchors)

        for i in range(10): session.run(stepio)

        start = time.perf_counter()

        session.run(stepio)

        end = time.perf_counter()
        elapsed = end - start
        timePerEvent = elapsed / n
        print("{:.12f}".format(elapsed))
