__author__ = "Hammad Aslam Khan"
__copyright__ = "Copyright 2019, Inferencing Pipeline for ONNX Neural Networks"
__credits__ = ["Hammad"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Hammad"
__email__ = "raohammad@gmail.com"
__status__ = "NA"

import abc

class SourceBase(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def __init__(self): 
        print('SourceBase initializer called.') 
  
    @abc.abstractmethod
    def delegate(self, args, callback):
        """receives args and a callback function and invokes that function 
        whenever data is received
        """

    @abc.abstractmethod
    def __del__(self):
        print('SourceBase destructor called')