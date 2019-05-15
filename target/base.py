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

class TargetBase(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def __init__(self): 
        print('TargetBase initializer called.') 
  
    @abc.abstractmethod
    def dumpData(self, nnDataBase):
        """receives data in this dumpData function. Every target will handle dumped data in its own way
        whenever data is received. Inherited class must be segregated depending on subclass of NNDataBase base class
        """

    @abc.abstractmethod
    def __del__(self):
        print('SourceBase destructor called')