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
from kafka import KafkaProducer
from kafka.errors import KafkaError

class KafkaTarget(TargetBase):
    def __init__(self, args):
        self.name = 'target.KafkaTarget'
        self.ip = args.ip
        self.port = args.port
        self.topicout = args.topicout
        self.producer = KafkaProducer(bootstrap_servers=[self.ip+':'+self.port])
        self.future = self.producer.send(self.ip, b'row_bytes')
        print('KafkaTarget initializer called')

    #function dumpData will have to handle data depending on data type i.e. subsclasses of NNDataBase
    def dumpData(self, nnDataBase):
        if isinstance(nnDataBase, NNImageData):
            print('dumping received data to kafka output topic:'+self.topicout)
            responseId, data, args = nnDataBase.nnData()
            print('requestId:'+responseId if responseId is not None else 'None'+' data:'+data.decode("utf-8"))
            self.producer.send(self.topicout, 
                key=b'(responseId if responseId is not None else "None")', 
                value=bytes(args.result, 'utf-8'))
        #if isinstance(someOtherInstance, SomeOtherClass):...
        else:
            print('data received at '+self.name+' is of unidentified format')
        return super().dumpData(nnDataBase)
        #for example for kafka, it shall trigger the producter and for DB it would be direct DB storage
        
    def __del__(self):
        print('KafkaTarget destructor called')
        self.producer.close()
        return super().__del__()
