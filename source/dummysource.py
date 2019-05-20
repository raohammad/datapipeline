__author__ = "Hammad Aslam Khan"
__description__ = "Data pipeline for Inferencing on ONNX Neural Networks"
__copyright__ = "Copyright 2019"
__credits__ = ["Hammad"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Hammad"
__email__ = "raohammad@gmail.com"
__status__ = "NA"

import abc
import time
from source.base import SourceBase
from common.args import Args

class DummySource(SourceBase):
    def __init__(self, args):
        self.name = 'source.DummySource'
        self.label = args.label
        self.name = args.name
        self.attribute = args.attribute
        print('DummySource initializer called ')

    #delegate method is used by nn, when execution starts, to delegate with callback
    def delegate(self, args, callback):
        print('delegating to the callback method')
        #for example for kafka, start the consumer here that listens on certain topic
        newArgs = Args()
        newArgs.timestamp = time.time()
        newArgs.requestid = 'abc'
        newArgs.data = 'data'
        callback(newArgs)

    def __del__(self):
        print('SourceTemplate destructor called')
        return super().__del__()