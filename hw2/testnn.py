#!/usr/bin/env python

"""
testnn.py -- unit tests for artificial neural nets implemented in nn.py
"""

import functools
import math
import random
import unittest

import nn

def logit(dblP):
    dblP = float(dblP)
    return math.log(dblP/(1.0 - dblP))

def sigmoid_prime_from_sigmoid(dblSigmoid):
    return dblSigmoid*(1.0 - dblSigmoid)

DEFAULT_REPEAT = 100

def repeated(fxn):
    @functools.wraps(fxn)
    def wrapper(obj, *args, **kwargs):
        cRepeat = obj.REPEAT if hasattr(obj, "REPEAT") else DEFAULT_REPEAT
        for _ in xrange(cRepeat):
            fxn(obj, *args, **kwargs)
    return wrapper

def randlist(dblLo, dblHi, cNumbers):
    dblDiff = dblHi - dblLo
    return [dblDiff*(random.random()) + dblLo for _ in xrange(cNumbers)]

def build_pcpt(cInputs,dblLo=1.0,dblHi=-10):
    return nn.Perceptron(cInputs, randlist(dblLo,dblHi,cInputs))

def pairwise_factors(listDbl, dblLo=-3.0, dblHi=3.0):
    listRandom = randlist(dblLo, dblHi, len(listDbl))
    listDblQuotient = []
    for dblOrig,dblRandom in zip(listDbl, listRandom):
        listDblQuotient.append(dblOrig/dblRandom)
    return listRandom,listDblQuotient

def randlist_for_sum(dblSum, cSize, dblSpread):
    if cSize < 1:
        raise ValueError("random list must be of length 1 or more.")
    dblRunningSum = 0.0
    listRandom = []
    dblBase = dblSum/float(cSize)
    for _ in xrange(cSize-1):
        dbl = random.random()*dblSpread + dblBase
        dblRunningSum += dbl
        listRandom.append(dbl)
    listRandom.append(dblSum - dblRunningSum)
    return listRandom

def build_net(listCLayer):
    listLayer = []
    for cInput,cPcpt in zip(listCLayer[:-1],listCLayer[1:]):
        listPcpt = []
        for ixPcpt in xrange(cPcpt):
            listDblW = randlist(-1.0, 1.0, cInput)
            listPcpt.append(nn.Perceptron(listDblW,0.0,ixPcpt))
        listLayer.append(nn.NeuralNetLayer(cInput,listPcpt))
    return nn.NeuralNet(listCLayer[0], listLayer)

class MathTest(unittest.TestCase):
    REPEAT = 1000

    def scales_linearly(self, dblM, dblB, listDbl, fxn):
        for dbl in listDbl:
            dblResult = fxn(dbl)
            dblExpected = dblM*dbl + dblB
            self.assertAlmostEqual(dblExpected, dblResult)
    
    @repeated
    def test_sigmoid(self):
        dblP = random.random()
        self.assertAlmostEqual(dblP, nn.sigmoid(logit(dblP)))

    @repeated
    def test_pcpt_activation(self):
        cInput = random.randint(5,10)
        dblW0 = random.random()*0.5
        listDblTarget = randlist(-1.0, 1.0, cInput)
        listDblX,listDblInput = pairwise_factors(listDblTarget,-1.0,1.0)
        pcpt = nn.Perceptron(listDblX, dblW0, 0)
        dblActivation = nn.pcpt_activation(pcpt,listDblInput)

        dblExpected = sum(listDblTarget)
        dblActual = logit(dblActivation) - dblW0
        self.assertAlmostEqual(dblExpected, dblActual, 4)

    @repeated
    def test_output_error(self):
        dblSum = random.random()*1000.0
        dblTarget = random.random()*1000.0
        dblResult = nn.output_error(dblTarget, dblSum)
        self.assertAlmostEqual(dblSum, dblResult + dblTarget)

    @repeated
    def test_hidden_error(self):
        cLayer = random.randint(10,15)
        cNextLayer = random.randint(10,15)
        ixPcpt = random.randint(0,cLayer-1)
        listDblExpected = randlist(-3.0,3.0,cNextLayer)
        listDblTarget,listDblDelta = pairwise_factors(listDblExpected)
        listPcpt = []
        for ix in xrange(cNextLayer):
            listDblW = randlist(-3.0,3.0,cLayer)
            listDblW[ixPcpt] = listDblTarget[ix]
            listPcpt.append(nn.Perceptron(listDblW, random.random()*100.0, ix))
        layer = nn.NeuralNetLayer(cLayer, listPcpt)
        pcpt = nn.Perceptron(randlist(-1.0,1.0,cLayer), 0.0, ixPcpt)
        dblResult = nn.hidden_error(listDblDelta, pcpt, layer)
        self.assertAlmostEqual(sum(listDblExpected), dblResult, 4)

    @repeated
    def test_compute_delta(self):
        dblActivation = random.random()
        dblError = 2.0*random.random() - 1.0
        dblDelta = nn.compute_delta(dblActivation, dblError)
        dblActivationPrime = sigmoid_prime_from_sigmoid(dblActivation)
        self.assertAlmostEqual(dblError, dblDelta/dblActivationPrime)
        self.assertAlmostEqual(dblActivationPrime, dblDelta/dblError)

    @repeated
    def test_update_weight(self):
        dblW,dblLR,dblIn,dblDelta = randlist(-1.0,1.0,4)
        def test_ix(ix):
            def wrapper(dbl):
                listArgs = [dblW,dblLR,dblIn,dblDelta]
                listArgs[ix] = dbl
                return nn.update_weight(*listArgs)
            return wrapper
        listFloats = [float(i) for i in xrange(-5,5)]
        self.scales_linearly(1.0,dblLR*dblIn*dblDelta,listFloats,test_ix(0))
        self.scales_linearly(dblIn*dblDelta,dblW,listFloats,test_ix(1))
        self.scales_linearly(dblLR*dblDelta,dblW,listFloats,test_ix(2))
        self.scales_linearly(dblLR*dblIn, dblW, listFloats,test_ix(3))

class PerceptronTest(unittest.TestCase):
    REPEAT = 100

    @repeated
    def test_update_pcpt(self):
        dblLearningRate = random.random() + 0.5
        cWeight = random.randint(10,20)
        listDblInput = randlist(-0.5,0.5,cWeight)
        listDblW = randlist(-0.5,0.5,cWeight)
        dblW0 = random.random() - 0.5
        dblActivation = random.random()*0.5 + 0.25
        dblError = random.random() - 0.5
        dblDelta = nn.compute_delta(dblActivation,dblError)
        ixPcpt = random.randint(0,100)
        pcpt = nn.Perceptron(list(listDblW), dblW0, ixPcpt)
        nn.update_pcpt(pcpt, listDblInput, dblDelta,  dblLearningRate)
        self.assertEqual(len(listDblW), len(pcpt.listDblW))
        dblProductBase = dblLearningRate*dblActivation*dblError
        for dblW,dblIn,dblWOrig in zip(pcpt.listDblW,listDblInput,listDblW):
            dblProduct = dblProductBase*dblIn
            dblExpected = dblProduct - (dblW - dblWOrig)
            self.assertAlmostEqual(dblProduct*dblActivation, dblExpected)
        self.assertAlmostEqual(dblProductBase*dblActivation,
                               dblProductBase - (pcpt.dblW0 - dblW0))

    @repeated
    def test_pcpt_activation(self):
        cInput = random.randint(10,50)
        listDblTarget = randlist(-0.05,0.05,cInput)
        listDblW,listDblInput = pairwise_factors(listDblTarget,-0.05,0.05)
        dblExpected = sum(listDblTarget)
        dblW0 = 0.1*(random.random() - 0.5)
        pcpt = nn.Perceptron(listDblW, dblW0, random.randint(0,1000))
        dblResult = nn.pcpt_activation(pcpt, listDblInput)
        self.assertAlmostEqual(dblExpected, logit(dblResult) - dblW0)

class NeuralNetTest(unittest.TestCase):
    REPEAT = 100
    
    @repeated
    def test_feed_forward_layer(self):
        listPcpt = []
        cInput = random.randint(5,10)
        cPcpt = random.randint(5,10)
        listDblTarget = randlist(-0.75, 0.75, cPcpt)
        listDblInput = randlist(-1.0,1.0,cInput)
        for ix,dblTarget in enumerate(listDblTarget):
            listDblProduct = randlist_for_sum(dblTarget, cInput+1, 0.5)
            listDblW = []
            for dblProduct,dblInput in zip(listDblProduct,listDblInput):
                listDblW.append(dblProduct/dblInput)
            listPcpt.append(nn.Perceptron(listDblW, listDblProduct[-1] ,ix))
        layer = nn.NeuralNetLayer(cInput, listPcpt)
        listDblOutput = nn.feed_forward_layer(layer, listDblInput)
        listDblLogit = [logit(dbl) for dbl in listDblOutput]
        for dblTarget,dblLogit in zip(listDblTarget,listDblLogit):
            self.assertAlmostEqual(dblTarget,dblLogit, 4)

    @repeated
    def test_build_layer_inputs_and_outputs(self):
        listCLayer = [random.randint(2,4) for _ in xrange(4)]
        net = build_net(listCLayer)
        listDblInput = randlist(-1.0,1.0,net.cInputs)
        listIn,listOut = nn.build_layer_inputs_and_outputs(net, listDblInput)
        self.assertEqual(len(listIn),len(listOut))
        listZip = zip(listIn,listOut,listCLayer,listCLayer[1:],net.listLayer)
        for listDblIn,listDblOut,cSizeIn,cSizeOut,layer in listZip:
            self.assertEqual(cSizeIn, len(listDblIn))
            self.assertEqual(cSizeOut, len(listDblOut))
            self.assertEqual(listDblOut,nn.feed_forward_layer(layer,listDblIn))

    @repeated
    def test_layer_deltas(self):
        cInput = random.randint(10,20)
        listDblTarget = randlist(-3.0,3.0,cInput)
        listDblActivation,listDblError = pairwise_factors(listDblTarget)
        listDblDelta = nn.layer_deltas(listDblActivation,listDblError)
        listZip = zip(listDblDelta,listDblTarget,listDblActivation)
        for dblDelta,dblTarget,dblAct in listZip:
            self.assertAlmostEqual(dblTarget*(1.0 - dblAct),dblDelta, 5)

    @repeated
    def test_update_layer(self):
        cOutput = random.randint(5,10)
        cInput = random.randint(5,10)
        listDblOutputs = randlist(-1.0,1.0,cOutput)
        listDblError = []
        for dblOut in listDblOutputs:
            listDblError.append(1.0/(1.0 - dblOut))
        listDblInput = randlist(-1.0,1.0,cInput)
        listPcpt = []
        listListDblWOrig = []
        
        for ixPcpt in xrange(cOutput):
            listDblW = randlist(-1.0,1.0,cInput)
            listListDblWOrig.append(listDblW)
            listPcpt.append(nn.Perceptron(list(listDblW),0.0,ixPcpt))
        listDblDelta = nn.layer_deltas(listDblOutputs,listDblError)
        layer = nn.NeuralNetLayer(cInput,listPcpt)
        dblLearningRate = random.random()
        nn.update_layer(layer, listDblInput, listDblDelta, dblLearningRate)
        listZip = zip(layer.listPcpt,listListDblWOrig,listDblOutputs)
        for pcpt,listDblWOrig,dblOut in listZip:
            listZipInner = zip(pcpt.listDblW,listDblWOrig,listDblInput)
            for dblW,dblWOrig,dblIn in listZipInner:
                dblExpectedDiff = dblLearningRate*dblIn*dblOut
                self.assertAlmostEqual(dblExpectedDiff, dblW - dblWOrig, 4)

    @repeated
    def test_hidden_layer_error(self):
        cOutput = random.randint(5,10)
        cNextLayerSize = random.randint(5,10)
        dblSum = random.random() - 0.5
        dblDeltaSum = 3.0*(random.random() - 0.5)
        listPcptUpstream = [nn.Perceptron([], 0.0, ixPcpt)
                            for ixPcpt in xrange(cOutput)]
        listListDbl = []
        for ixUp in xrange(cOutput):
            listListDbl.append(randlist_for_sum(dblSum, cNextLayerSize, 2.0))
        listPcptDownstream = []
        for ixPcpt in xrange(cNextLayerSize):
            listDblW = [listDbl[ixPcpt] for listDbl in listListDbl]
            listPcptDownstream.append(nn.Perceptron(listDblW,0.0,ixPcpt))
        layerUp = nn.NeuralNetLayer(0,listPcptUpstream)
        layerDown = nn.NeuralNetLayer(cOutput,listPcptDownstream)
        listDblDelta = [1.0 for _ in xrange(cNextLayerSize)]
        listDblError = nn.hidden_layer_error(layerUp, listDblDelta, layerDown)
        self.assertAlmostEqual(dblSum*cOutput, sum(listDblError),5)

    def test_init_net(self):
        layer_sizes = [3, 2, 1]
        neural_net = nn.init_net(layer_sizes)
        self.assertEqual(2, len(neural_net.listLayer))
        # Make sure that the layers are configured correctly.
        layer0 = neural_net.listLayer[0]
        self.assertEqual(3, layer0.layer_input_size())
        self.assertEqual(2, layer0.layer_output_size())
        
        def check_perceptrons(layer, num_inputs):
            for p in layer.listPcpt:
                for w in p.listDblW:
                    self.assertTrue(w >= -0.1 and w <= 0.1)
                    self.assertTrue(p.dblW0 >= -0.1 and p.dblW0 <= 0.1)
                    self.assertEqual(num_inputs, len(p.listDblW))
        check_perceptrons(layer0, 3)

        # Weights on all inputs should be between -0.1 and 0.1
        layer1 = neural_net.listLayer[1]
        self.assertEqual(2, layer1.layer_input_size())
        self.assertEqual(1, layer1.layer_output_size())
        check_perceptrons(layer1, 2)

    def test_update_net(self):
        # Test a simple network to make sure that forward and backprop are
        # working properly.
        # 2 inputs, 2 hidden nodes, 1 output node
        net = nn.init_net([2, 2, 1])
        # Set the weights on the first hidden node to 0.1, the weights on the
        # second hidden node to -0.1.
        def init_weights(p, input_weight, w0):
          for i in xrange(len(p.listDblW)):
            p.listDblW[i] = input_weight
          p.dblW0 = w0
        # Weights for the first hidden node to be 0.1
        init_weights(net.listLayer[0].listPcpt[0], 0.1, 0.0)
        # Weights for the second hidden node to be -0.1
        init_weights(net.listLayer[0].listPcpt[1], -0.1, 0.0)
        # Weights for the output layer to be 0.1
        init_weights(net.listLayer[1].listPcpt[0], 1.0, 0.0)
        # Inputs are 1 and -1
        inst = nn.Instance(0, [ 1.0, -0.9 ])
        # Target output is 0.5
        targets = [ 0.5 ]
        # The output of hidden unit 1 will be 1 / (1 + e^(-0.01))
        # The output of hidden unit 2 will be 1 / (1 + e^(0.01))
        # The inputs to the output unit will be 1.0, leading to output 0.731
        # The error at the output will be -0.231
        nn.update_net(net, inst, 1.0, targets)
        def get_weight(layer_id, perceptron_id, input_id):
          if input_id == -1:
            return net.listLayer[layer_id].listPcpt[perceptron_id].dblW0
          return net.listLayer[layer_id].listPcpt[perceptron_id].listDblW[input_id]

        output = 1.0 / (1.0 + math.exp(-1.0))
        delta_out = (0.5 - output) * output * (1 - output)
        h1_out = 1.0 / (1.0 + math.exp(-.01))
        h2_out = 1.0 / (1.0 + math.exp(.01))
        self.assertAlmostEqual(1.0 + 1.0 * h1_out * delta_out, get_weight(1, 0, 0))
        self.assertAlmostEqual(1.0 + 1.0 * h2_out * delta_out, get_weight(1, 0, 1))
        self.assertAlmostEqual(1.0 * 1.0 * delta_out, get_weight(1, 0, -1))

        in1 = 1.0 / (1.0 + math.exp(-0.01))
        delta_hidden1 = in1 * (1 - in1) * delta_out
        # For the hidden units, in is 0, so no weight updates should occur.
        self.assertAlmostEqual(0.1 + 1.0 * delta_hidden1, get_weight(0, 0, 0))
        self.assertAlmostEqual(0.1 - 0.9 * delta_hidden1, get_weight(0, 0, 1))
        self.assertAlmostEqual(delta_hidden1, get_weight(0, 0, -1))

        in2 = 1.0 / (1.0 + math.exp(0.01))
        delta_hidden2 = in2 * (1 - in2) * delta_out
        self.assertAlmostEqual(-0.1 + 1.0 * delta_hidden2,
                               get_weight(0, 1, 0))
        self.assertAlmostEqual(-0.1 - 0.9 * delta_hidden2,
                               get_weight(0, 1, 1))
        self.assertAlmostEqual(delta_hidden2, get_weight(0, 0, -1))

class EncodingTest(unittest.TestCase):
    REPEAT = 1000
    
    def test_distributed_encode_label(self):
        for i in xrange(10):
            listDblEncoding = nn.distributed_encode_label(i)
            for ix,dblTarget in enumerate(listDblEncoding):
                if ix == i:
                    self.assertTrue(dblTarget > 0.8)
                else:
                    self.assertTrue(dblTarget < 0.2)

    @repeated
    def test_distributed_decode_net_output(self):
        listDblEncoding = []
        dblMax = None
        ixMax = None
        for ix in xrange(10):
            dbl = random.random()
            listDblEncoding.append(dbl)
            if dbl > dblMax:
                dblMax = dbl
                ixMax = ix
        iResult = nn.distributed_decode_net_output(listDblEncoding)
        self.assertEqual(ixMax, iResult)

    def test_binary_encode_label(self):
        for i in xrange(4):
            listDblEncoding = nn.binary_encode_label(i)
            iSum = 0
            for ix,dbl in enumerate(listDblEncoding):
                if dbl > 0.5:
                    iSum += 1 << ix
            self.assertEqual(iSum,i)

    @repeated
    def test_binary_decode_net_output(self):
        listDblEncoding = randlist(0.0,0.5,4)
        iLabel = random.randint(0,9)
        for ix in xrange(4):
            if (iLabel >> ix) & 0x1:
                listDblEncoding[ix] = 1.0 - (random.random()*0.5)
        iResult = nn.binary_decode_net_output(listDblEncoding)
        self.assertEqual(iLabel,iResult)
                
if __name__ == "__main__":
    unittest.main()
