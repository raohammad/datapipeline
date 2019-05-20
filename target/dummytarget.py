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
from target.base import TargetBase
from common.args import Args
from common.nndata import NNImageData

class DummyTarget(TargetBase):
    def __init__(self, args):
        self.name = 'target.DummyTarget'
        self.label = args.label
        self.name = args.name
        self.attribute = args.attribute
        print('DummyTarget initializer called')

    #function dumpData will have to handle data depending on data type i.e. subsclasses of NNDataBase
    def dumpData(self, nnDataBase):
        print('dumping received data to screen')
        if isinstance(nnDataBase, NNImageData):
            requestId, data, args = nnDataBase.nnData()
            print('requestId:'+requestId if requestId is not None else 'None'+' data:'+data.decode("utf-8"))
        #if isinstance(someOtherInstance, SomeOtherClass):...
        else:
            print('data received at '+self.name+' is of unidentified format')
        return super().dumpData(nnDataBase)
        #for example for kafka, it shall trigger the producter and for DB it would be direct DB storage
        
    def __del__(self):
        print('TargetTemplate destructor called')
        return super().__del__()