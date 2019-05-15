__author__ = "Hammad Aslam Khan"
__description__ = "Pipeline design pattern | Create a data pipeline to trigger a Neural network in order, data source --> NN prediction --> data target"
__copyright__ = "Copyright 2019, Inferencing Pipeline for ONNX Neural Networks"
__credits__ = ["Hammad"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Hammad"
__email__ = "raohammad@gmail.com"
__status__ = "NA"

# Driver Program, essentially below code is part of application that invikes the pipeline #
from source.sourcetemplate import SourceTemplate
from target.targettemplate import TargetTemplate
from nn.nntemplate import NNTemplate
from common.args import Args

sourceArgs = Args()
sourceArgs.ip = '192.168.1.144'
sourceArgs.port = '9092'
sourceArgs.user = 'test'
sourceArgs.password = 'test'

targetArgs = Args()
targetArgs.ip = '192.168.1.144'
targetArgs.port = '9092'
targetArgs.user = 'hello'
targetArgs.password = 'hello'

nnArgs = Args()
nnArgs.ip = '192.168.1.144'
nnArgs.port = '9092'
nnArgs.user = 'hello'
nnArgs.password = 'hello'

nnTemplate = NNTemplate(nnArgs)

#SourceTemplate and TargetTemplate are simple placeholders
nnTemplate.execute(SourceTemplate(sourceArgs), TargetTemplate(targetArgs))
