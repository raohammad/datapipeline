__author__ = "Hammad Aslam Khan"
__copyright__ = "Copyright 2019, Data Pipeline for Inferencing on ONNX Neural Networks"
__credits__ = ["Hammad"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Hammad"
__email__ = "raohammad@gmail.com"
__status__ = "NA"

import abc
import time
from target.base import TargetBase
from common.args import Args

class TargetTemplate(TargetBase):
    def __init__(self, args):
        self.name = 'template'
        self.ip = args.ip
        self.port = args.port
        self.user = args.user
        self.password = args.password
        print('TargetTemplate initializer called')

    #function 
    def dumpData(self, args):
        print('dumping received data to screen:'+args.result)
        #for example for kafka, it shall trigger the producter and for DB it would be direct DB storage
        
    def __del__(self):
        print('TargetTemplate destructor called')
        return super().__del__()]