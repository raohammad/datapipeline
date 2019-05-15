__author__ = "Hammad Aslam Khan"
__description__ = "Data pipeline for Inferencing on ONNX Neural Networks"
__copyright__ = "Copyright 2019"
__credits__ = ["Hammad"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Hammad"
__email__ = "raohammad@gmail.com"
__status__ = "NA"

from nn.base import NNBase
from common.args import Args
import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
from mxnet.gluon.data.vision import transforms
from mxnet.contrib.onnx.onnx2mx.import_model import import_model
import os

class NNResNet50v2(NNBase):

    def __init__(self, args):
        self.args = args
        print('NNResNet50v2 initializer called')
        print('NNResNet50v2 model being loaded')
        model_path= 'onnx/resnet50v2.onnx'
        sym, arg_params, aux_params = import_model(model_path)
        self.Batch = namedtuple('Batch', ['data'])
        self.mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        self.mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
                label_shapes=mod._label_shapes)
        self.mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
        with open('onnx/synset.txt', 'r') as f:
            self.labels = [l.rstrip() for l in f]
        print('NNResNet50v2 model loading done')
        print('NNTemplate initializer called')
    
    def execute(self, sourceBase, targetBase):
        self.sourceBase = sourceBase
        self.targetBase = targetBase
        self.sourceBase.delegate(self.args, self.callback)

    def get_image(sourceimage, show=False):
        img = mx.img.imdecode(sourceimage)
        if img is None:
            return None
        if show:
            plt.imshow(img.asnumpy())
            plt.axis('off')
        return img

    def preprocess(self, img):
        transform_fn = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform_fn(img)
        img = img.expand_dims(axis=0)
        return img

    def predict(self, img):
        img = preprocess(img)
        self.mod.forward(self.Batch([img]))
        # Take softmax to generate probabilities
        scores = mx.ndarray.softmax(self.mod.get_outputs()[0]).asnumpy()
        # print the top-5 inferences class
        scores = np.squeeze(scores)
        a = np.argsort(scores)[::-1]
        for i in a[0:5]:
            print('class=%s ; probability=%f' %(self.labels[i],scores[i]))

    def callback(self, nnImageData):
        print('callback of NNTemplate called with args:')
        requestId, data, args = nnImageData.nnData()
        result = predict(data['image'])
        #print('requestId:'+requestId if requestId is not None else 'None'+' data:'+data.decode("utf-8"))
        self.targetBase.dumpData(nnImageData)
        return super().callback(nnImageData)

    def __del__(self):
        return super().__del__()
