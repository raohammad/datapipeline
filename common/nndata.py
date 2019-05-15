from common.args import Args
import abc

class NNDataBase(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """NNDataBase initializer called"""

    @abc.abstractmethod
    def nnData(self):
        """method that should be implemented by all nn data classes"""
    
    @abc.abstractmethod
    def __del__(self):
        """NNDataBase distructor called"""

#for NNs working on individual images, same data class is used
#valid for all image all below mentioned NNs that are trained on ImageNet
#SqueezeNet
#VGG
#ResNet [all versions]
#MobileNet

class NNImageData(NNDataBase):
    def __init__(self, requestId, data, args ):
        self.requestId = requestId
        self.data = data
        self.args = args
        """NNImageData initializer called"""

    def nnData(self):
        return self.requestId, self.data, self.args

    def __del__(self):
        print('NNData destructor called')
        """NNImageData destructor called"""
