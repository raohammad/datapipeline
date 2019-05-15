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

class NNTemplate(NNBase):
    
    def __init__(self, args):
        self.args = args
        print('NNTemplate initializer called')
    
    def execute(self, sourceBase, targetBase):
        self.sourceBase = sourceBase
        self.targetBase = targetBase
        self.sourceBase.delegate(self.args, self.callback)

    def preprocess(self, args):
        return super().preprocess(args)

    def predict(self, args):
        print('prediction called on NNTemplate')
        return 'prediction result'

    def callback(self, args):
        print('callback of NNTemplate called with args:')
        args.result = self.predict(self.args)
        print(args)
        self.targetBase.dumpData(args)
        return super().callback(args)

    def __del__(self):
        return super().__del__()
