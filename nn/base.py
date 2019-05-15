__author__ = "Hammad Aslam Khan"
__copyright__ = "Copyright 2019, Inferencing Pipeline for ONNX Neural Networks"
__credits__ = ["Hammad"]
__license__ = "GPL"
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
    @abc.abstractmethod
    def execute(self, args):
        """ every NN's entry point. args.SourceBase and args.TargetBase are passed. Always SourceBase.deletate function will take the callback method 
        of this class
        """

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
    def callback(self, args):
        """Save the data object to the output."""

    @abc.abstractmethod
    def __del__(self):
        print('CVNNBase destructor called')