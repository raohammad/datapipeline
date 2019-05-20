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
import json
import cv2
import base64

class NNResNet34v1(NNBase):

    def __init__(self, args):
        self.args = args
        print('NNResNet50v2 initializer called')
        print('NNResNet50v2 model being loaded')
        model_path= 'nn/onnx/resnet50v2.onnx'
        sym, arg_params, aux_params = import_model(model_path)
        if len(mx.test_utils.list_gpus())==0:
            ctx = mx.cpu()
        else:
            ctx = mx.gpu(0)
        self.Batch = namedtuple('Batch', ['data'])
        self.img_path = 'nn/onnx/file.jpg'
        self.mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        self.mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], 
                label_shapes=self.mod._label_shapes)
        self.mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
        with open('nn/onnx/synset.txt', 'r') as f:
            self.labels = [l.rstrip() for l in f]
        print('NNResNet50v2 model loading done')
        print('NNTemplate initializer called')

    def get_image(img_path, show=False):
        img_path = 'nn/onnx/file.jpg'
        img = mx.image.imread(img_path)
        if img is None:
            return None
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

    def predict(self, img_path):
        imgr = self.get_image(img_path)
        img = self.preprocess(imgr)
        self.mod.forward(self.Batch([img]))
        # Take softmax to generate probabilities
        scores = mx.ndarray.softmax(self.mod.get_outputs()[0]).asnumpy()
        # print the top-5 inferences class
        scores = np.squeeze(scores)
        a = np.argsort(scores)[::-1]
        jsonout = {}
        jsonout['classes'] = []
        for i in a[0:5]:
            print('class=%s ; probability=%f' %(self.labels[i],scores[i]))
            jsonout['classes'].append({'class':self.labels[i], 'probability':str(scores[i])})
        return jsonout

    def callback(self, nnImageData):
        print('callback of NNTemplate called with args:')
        requestId, data, args = nnImageData.nnData()
        jsondata = json.loads(data.decode("utf-8"))
        print(jsondata['requestId'])
        if jsondata['img'] is not None:
            filedata = base64.b64decode(jsondata['img'])
            with open(self.img_path, 'wb') as f_output:
                f_output.write(filedata)
            jsondata['img'] #please note this is in base64 format, it needs to be converted back to image
            result = self.predict(self.img_path)
            #print('requestId:'+requestId if requestId is not None else 'None'+' data:'+data.decode("utf-8"))
            nnImageData.args.result = json.dumps(result);
            for t in self.targetBase:
                t.dumpData(args)
        return super().callback(nnImageData)

    def __del__(self):
        return super().__del__()