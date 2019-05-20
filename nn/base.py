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

class NNBase(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def __init__(self, args): 
        """load the relevant ONNX model
        """
    #receives source list and  target list
    def execute(self, sourceBase, targetBase):
        self.sourceBase = sourceBase
        self.targetBase = targetBase
        for s in self.sourceBase:
            s.delegate(self.args, self.callback)

    @abc.abstractmethod
    def preprocess(self, args):
        """ process the data whenever received 
        the method is invokved by the callback method of same class
        """

    @abc.abstractmethod
    def predict(self, args):
        """ apply the NN model whenever prediction needs to be made
        """

    @abc.abstractmethod
    def callback(self, nnDataBase):
        """Save the data object to the output. Always accept a subclass of common.nndata.NNDataBase"""

    @abc.abstractmethod
    def __del__(self):
        print('CVNNBase destructor called')