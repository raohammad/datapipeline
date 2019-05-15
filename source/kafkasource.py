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
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from common.nndata import NNImageData
import time
import argparse 

class KafkaSource(SourceBase):
    def __init__(self, args):
        self.name = 'source.KafkaSource'
        self.ip = args.ip
        self.port = args.port
        self.topicin = args.topicin
        self.consumer= KafkaConsumer(self.topicin, group_id = None, bootstrap_servers=[self.ip+':'+self.port])
        print('KafkaSource initializer called')

    #delegate method is used by nn, when execution starts, to delegate with callback
    def delegate(self, args, callback):
        print('delegating to the callback method')
        #sends message whenever message is received
        for message in self.consumer:
            print("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition, message.offset, message.key, message.value))
            newArgs = Args() #other args
            newArgs.name = self.name
            newArgs.timestamp = time.time()
            newArgs.requestid = 'abc'
            newArgs.topic = message.topic
            newArgs.partition = message.partition
            newArgs.offset = message.offset
            newArgs.key = message.key
            newArgs.value = message.value
             
            nnDataBase = NNImageData(message.key, message.value, newArgs) #key and value are standard data for this
            callback(nnDataBase)

    def __del__(self):
        print('SourceTemplate destructor called')
        self.consumer.close()
        return super().__del__()