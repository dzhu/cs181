#!/usr/bin/env python

"""
tasknn.py -- Visualizations for neural networks.
"""

from os import path
import random

from tfutils import tftask

import nn

XOR_INSTANCES = [nn.Instance(0.1, [-1.0,-1.0]), nn.Instance(0.9, [-1.0,1.0]),
                 nn.Instance(0.9, [1.0,-1.0]), nn.Instance(0.1, [1.0,1.0])]

def build_xor_net():
    HIDDEN_NODES = 2
    ROUNDS = 5000
    LEARNING_RATE = 0.35
    assert XOR_INSTANCES
    net = nn.init_net([2, HIDDEN_NODES, 1], 0.001)
    for ixRound in xrange(ROUNDS):
        dblAlpha = 2.0*ROUNDS/(ixRound + ROUNDS)
        for inst in XOR_INSTANCES:
            nn.update_net(net, inst, dblAlpha, [inst.iLabel])
    return net

def serialize_net(net):
    def build_edge(sLabelIn, sLabelOut, dblWeight):
        return (sLabelIn, sLabelOut,
                {"weight": edge_weight(dblWeight),
                 "color": "#000000" if dblWeight > 0 else "#FF0000"})
    def edge_weight(dblWeight):
        return min(4.0, max(dblWeightScale*abs(dblWeight), 0.5))
    listEdge = []
    dblWeightScale = 10.0
    sNodeTmpl = "Node (%d, %d)"
    for ixInput in xrange(net.listLayer[0].cInputs):
        for pcpt in net.listLayer[0].listPcpt:
            dblWeight = pcpt.listDblW[ixInput]
            listEdge.append(build_edge("Input %d" % ixInput, sNodeTmpl
                                       % (0,pcpt.ix), dblWeight))
    listPairLayer = zip(net.listLayer[:-1],net.listLayer[1:])
    for ixLayer,(layerIn,layerOut) in enumerate(listPairLayer):
        for ixPcptIn,pcptIn in enumerate(layerIn.listPcpt):
            sLabelIn = sNodeTmpl % (ixLayer,ixPcptIn)
            for ixPcptOut,pcptOut in enumerate(layerOut.listPcpt):
                sLabelOut = sNodeTmpl % (ixLayer+1,ixPcptOut)
                dblWeight = pcptOut.listDblW[ixPcptIn]
                listEdge.append(build_edge(sLabelIn,sLabelOut,dblWeight))
    return listEdge

def evaluate_net(net,listInst,fxnDecode):
    cCorrect = 0
    for inst in listInst:
        iResult = fxnDecode(nn.feed_forward(net,inst.listDblFeatures))
        cCorrect += int(iResult == inst.iLabel)
    return float(cCorrect)/float(len(listInst))

def build_and_measure_net(net,listInstTrain,listInstTest,
                          fxnEncode,fxnDecode,dblLearningRate,
                          cRounds):
    for _ in xrange(cRounds):
        for inst in listInstTrain:
            listDblTarget = fxnEncode(inst.iLabel)
            nn.update_net(net, inst, dblLearningRate, listDblTarget)
        dblTestError = evaluate_net(net, listInstTest, fxnDecode)
        dblTrainingError = evaluate_net(net, listInstTrain, fxnDecode)
        yield dblTestError,dblTrainingError

def data_filename(sFilename):
    return path.join(path.dirname(__file__), path.basename(sFilename))

TRAINING_9K = "training-9k.txt"
TEST_1K = "test-1k.txt"

def load_instances(sFilename, cMaxInstances=None):
    return nn.load_data(data_filename(sFilename), cMaxInstances)

def load_training_9k(cMaxInstances):
    return load_instances(TRAINING_9K,cMaxInstances)

def load_test_1k(cMaxInstances):
    return load_instances(TEST_1K,cMaxInstances)

def performance_graph(listListDblResults, sTitle):
    if len(listListDblResults) == 1:
        listSName = ["Accuracy"]
    else:
        listSName = ["Test Accuracy", "Training Accuracy",
                     "Validation Accuracy"]
    listDictSeries = []
    for sName,listDblResult in zip(listSName,listListDblResults):
        listDictSeries.append({"name": sName, "data": listDblResult})
    return {"chart": {"defaultSeriesType": "line"},
            "xAxis": {"title": {"text": "Round"}},
            "yAxis": {"title": {"text": "Accuracy"}},
            "title": {"text": sTitle},
            "series": listDictSeries}

class SigmoidTask(tftask.ChartTask):
    def get_name(self):
        return "Plot A Sigmoid Curve"
    def get_description(self):
        return ("Generate a plot the activation function for the perceptron"
                " nodes which will make up your neural network.")
    def get_priority(self):
        return -1
    def task(self):
        iScale = 10
        iExtent = 5
        iMin = -iScale*iExtent
        iMax = iScale*iExtent
        listDblX = [float(i)/float(iScale) for i in xrange(iMin,iMax)]
        listDblSigmoid = [nn.sigmoid(dblX) for dblX in listDblX]
        listPairData = zip(listDblX,listDblSigmoid)
        return {"chart": {"defaultSeriesType": "line"},
                "title": {"text": "Sigmoid Function"},
                "xAxis": {"title":{"text": "Perceptron Input"},
                          "min": -iExtent, "max": iExtent},
                "yAxis": {"title":{"text":"Activation"}, "min":0.0,
                          "max": 1.0},
                "series": [{"name":"Activation", "data": listPairData}]}
    
class XorDisplayTask(tftask.GraphTask):
    def get_name(self):
        return "Solve XOR: Part I"
    def get_description(self):
        return ("Build an artificial neural network that learns the XOR "
                "function from training data.")
    def get_prirority(self):
        return 0
    def task(self):
        net = build_xor_net()
        return serialize_net(net)

class XorClassifyTask(tftask.ChartTask):
    def get_name(self):
        return "Solve XOR: Part II"
    def get_description(self):
        return ("Generate a scatter plot demonstrating how your "
                "neural network from Part I correctly solves XOR."
                " Black and red indicate true and false labels, respectively,"
                " while upward arrows indicate positive classification, and"
                " downward arrows indicate negative classification.")
    def get_priority(self):
        return 1
    def task(self):
        net = build_xor_net()
        pts = []
        listTrue = []
        listFalse = []
        listSMarker = ["triangle-down", "triangle"]
        listSColor = ["#FF0000", "#000000"]
        for inst in XOR_INSTANCES:
            dblOut = nn.feed_forward(net, inst.listDblFeatures)[0]
            fLabel = inst.iLabel > 0.5
            iGuess = int(dblOut > 0.5)
            dblX,dblY = inst.listDblFeatures
            dictPoint = {"x": dblX, "y": dblY,
                         "marker": {"symbol": listSMarker[iGuess],
                                    "fillColor": listSColor[int(fLabel)]}}
            if fLabel:
                listTrue.append(dictPoint)
            else:
                listFalse.append(dictPoint)
        return {"chart": {"defaultSeriesType":"scatter"},
                "plotOptions": {"scatter": {
                    "marker": {"radius": 20,
                               "states": {
                                   "hover":{"enabled":False}}}}},
                "title": {"text": "XOR Classification"},
                "xAxis": {"title": {"text": "Input 0"},
                          "min": -1.05, "max": 1.05, "tickInterval": 1,
                          "gridLineWidth": 1},
                "yAxis": {"title": {"text": "Input 1"},
                          "min": -1.2, "max": 1.2, "tickInterval": 1,
                          "startOnTick": False, "endOnTick": False},
                "series": [{"name":"True", "data": listTrue},
                           {"name": "False", "data": listFalse}],
                "legend": {"enabled": False},
                "tooltip": {"enabled": False}}

class DigitWarmup(tftask.ChartTask):
    ROUNDS = 10
    LEARNING_RATE = 0.5
    def get_name(self):
        return "Digit Recognition Warmup"
    def get_description(self):
        return ("Train an artificial neural network to recognize handwritten"
                " digits using ten training instances over three rounds."
                " Run this task to make sure everything works before running"
                " the longer training tasks. Do not expect good accuracy"
                " on this task.")
    def get_priority(self):
        return 3
    def task(self):
        listInst = load_training_9k(10)
        net = nn.init_net([14*14,15,10])
        listDblResult = list(build_and_measure_net(
            net, listInst, listInst, nn.distributed_encode_label,
            nn.distributed_decode_net_output, self.LEARNING_RATE,
            self.ROUNDS))
        return performance_graph([[a for a,_ in listDblResult]],
                                 "Digit Recognition Training Accuracy")

class DigitClassificationDistributed(tftask.ChartTask):
    LEARNING_RATE = 0.5
    ROUNDS = 10
    TRAINING_INSTANCES = 9000
    TEST_INSTANCES = 1000
    NETWORK_CONFIGURATION = [14*14,15,10]
    def get_name(self):
        return "Digit Classification (Distributed Encoding)"
    def get_description(self):
        return ("Train an artificial neural network to classify handwritten"
                " digits. This task operates on a subset of the full training"
                " data provided in this assignment.")
    def get_priority(self):
        return 4
    def measure_performance(self, fxnEncode, fxnDecode):
        listInstTraining = load_training_9k(self.TRAINING_INSTANCES)
        listInstTest = load_test_1k(self.TEST_INSTANCES)
        net = nn.init_net(self.NETWORK_CONFIGURATION)
        listDblResult = list(build_and_measure_net(
            net,listInstTraining,listInstTest, fxnEncode, fxnDecode,
            self.LEARNING_RATE, self.ROUNDS))
        listDblTest = [a for a,_ in listDblResult]
        listDblTrain = [b for _,b in listDblResult]
        sTitle = ("Digit Recognition Test Accuracy Trained on %d Instances"
                  % len(listInstTraining))
        return performance_graph([listDblTest,listDblTrain], sTitle)
    def task(self):
        return self.measure_performance(nn.distributed_encode_label,
                                        nn.distributed_decode_net_output)

class DigitClassificationBinary(DigitClassificationDistributed):
    NETWORK_CONFIGURATION = [14*14,15,4]
    def get_name(self):
        return "Digit Classification (Binary Encoding)"
    def get_description(self):
        return "Train a network as above, but with a binary output encoding."
    def get_priority(self):
        return 5
    def task(self):
        return self.measure_performance(nn.binary_encode_label,
                                        nn.binary_decode_net_output)

class DigitClassificationThirty(DigitClassificationDistributed):
    NETWORK_CONFIGURATION = [14*14, 30, 10]
    TRAINING_INSTANCES = 3000
    ROUNDS = 5
    def get_name(self):
        return "Digit Classification (30 Hidden Units)"
    def get_description(self):
        return "Train and evaluate a network with thirty hidden units."
    def get_priority(self):
        return 6

def main(argv):
    return tftask.main()

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
