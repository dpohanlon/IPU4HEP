import torch
import numpy as np
try:
    import popart
except:
    pass
import time
import argparse

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

cuda = torch.cuda.is_available()
device = "cuda:0" if cuda else "cpu"

def convOut(input_length, k, pad = 0):
    return input_length + 2 * pad - (k - 1)

class NetCNN(torch.nn.Module):
    def __init__(self, hidden_size, input_size, input_length, k1, k2, pool):
        super(NetCNN, self).__init__()

        pad = 0
        conv1Out = convOut(input_length, k1)
        conv2Out = convOut(conv1Out, k2)

        self.relu = torch.nn.functional.relu

        self.conv1 = torch.nn.Conv2d(input_size, hidden_size, kernel_size = [k1,1])
        self.conv2 = torch.nn.Conv2d(hidden_size, hidden_size, kernel_size = [k2,1])

        self.maxpool = torch.nn.MaxPool2d((pool, 1))
        self.dropout = torch.nn.Dropout(0.5)
        self.batchNorm = torch.nn.BatchNorm2d(conv1Out * hidden_size,1)
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear((conv2Out * hidden_size) // pool, hidden_size,1)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linearOut = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        debug = False

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)

        return self.linearOut(x)

    def eval(self, x):
        return torch.sigmoid(self.forward(x))

if __name__ == '__main__':

    argParser = argparse.ArgumentParser()

    argParser.add_argument("-n", type = int, dest = "n", default = 256, help = 'Batch size.')
    argParser.add_argument("-t", type = int, dest = "nTracks", default = 256, help = 'Number of tracks.')
    argParser.add_argument("-f", type = int, dest = "nFeatures", default = 18, help = 'Number of features.')
    argParser.add_argument("--k1", type = int, dest = "k1", default = 2, help = 'Conv1 kernel window size.')
    argParser.add_argument("--k2", type = int, dest = "k2", default = 2, help = 'Conv2 kernel window size.')
    argParser.add_argument("--pool", type = int, dest = "pool", default = 2, help = 'Pooling width.')
    argParser.add_argument("-s", type = int, dest = "nHidden", default = 128, help = 'Hidden size of LSTM.')
    argParser.add_argument("-d", type = int, dest = "s", default = 1, help = 'Number of stacked conv layers.')
    argParser.add_argument("-c", default = False, action = "store_true", dest = 'cpu', help = 'Run on CPU.')
    argParser.add_argument("-g", default = False, action = "store_true", dest = 'gpu', help = 'Run on GPU.')
    args = argParser.parse_args()

    cpu = args.cpu # Make sure this runs on only one core, set OMP/MKL_NUM_THREADS=1
    gpu = args.gpu
    n = args.n
    nTracks = args.nTracks
    nFeatures = args.nFeatures
    nHidden = args.nHidden
    k1 = args.k1
    k2 = args.k2
    pool = args.pool

    onnx_model = 'cnn2d.onnx'

    # PYTORCH EXPORT

    # Padded with 1
    dummy_input = torch.randn((n, nFeatures, nTracks, 1),dtype=torch.float32)

    try:
        model = NetCNN(nHidden, nFeatures, nTracks, k1, k2, pool)
        if cuda and gpu:
            model = model.to(device)
            dummy_input = dummy_input.to(device)
    except:
        print('-1')
        exit(0)

    nParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(nParams)

    output_names = [ "tag" ]
    input_names = [ "data" ]

    torch.onnx.export(model, dummy_input, onnx_model, verbose=False, input_names=input_names, output_names=output_names)

    if cpu or gpu:

        start = time.perf_counter()

        model(dummy_input)

        end = time.perf_counter()

        elapsed = end - start
        timePerEvent = elapsed / n

        print("{:.12f}".format(elapsed))

    else:

        # POPART IMPORT

        graph_transformer = popart.GraphTransformer(onnx_model)

        anchors = {"tag" : popart.AnchorReturnType("ALL")}
        dataFeed = popart.DataFlow(1, anchors)
        device = popart.DeviceManager().acquireAvailableDevice(1)

        session = popart.InferenceSession(graph_transformer.getModelProto(), dataFeed, device)

        session.prepareDevice()

        inferenceAnchors = session.initAnchorArrays()

        inputs = np.random.rand(n, nFeatures, nTracks,1).astype(np.float32)

        stepio = popart.PyStepIO({"data": inputs}, inferenceAnchors)

        # for i in range(10): session.run(stepio)

        start = time.perf_counter()

        session.run(stepio)

        end = time.perf_counter()
        elapsed = end - start
        timePerEvent = elapsed / n
        print("{:.12f}".format(elapsed))
