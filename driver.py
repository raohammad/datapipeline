__author__ = "Hammad Aslam Khan"
__description__ = "Pipeline design pattern | Create a data pipeline to trigger a Neural network in order, data source --> NN prediction --> data target"
__copyright__ = "Copyright 2019"
__credits__ = ["Hammad"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Hammad"
__email__ = "raohammad@gmail.com"
__status__ = "NA"

# Driver Program, essentially below code is part of application that invikes the pipeline #
from source.sourcetemplate import SourceTemplate
from target.targettemplate import TargetTemplate
from source.kafkasource import KafkaSource
from target.kafkatarget import KafkaTarget
from nn.nntemplate import NNTemplate
from nn.nnRestNet50v2 import NNResNet50v2
from common.args import Args

# BEGIN Section: Based on Templates
#
# sourceArgs = Args()
# sourceArgs.ip = '192.168.1.144'
# sourceArgs.port = '9092'
# sourceArgs.user = 'test'
# sourceArgs.password = 'test'

# targetArgs = Args()
# targetArgs.ip = '192.168.1.144'
# targetArgs.port = '9092'
# targetArgs.user = 'hello'
# targetArgs.password = 'hello'

# nnArgs = Args()
# nnArgs.ip = '192.168.1.144'
# nnArgs.port = '9092'
# nnArgs.user = 'hello'
# nnArgs.password = 'hello'
#
# nnTemplate = NNTemplate(nnArgs)
# SourceTemplate and TargetTemplate are simple placeholders
# nnTemplate.execute(SourceTemplate(sourceArgs), TargetTemplate(targetArgs))
#
# END Section: Based on Templates


# BEGIN Section: Based on Kafka Source and Target or KafkaSource and TargetTemplate
kafkaSourceArgs = Args()
kafkaSourceArgs.ip = '192.168.1.144'
kafkaSourceArgs.port = '9092'
kafkaSourceArgs.topicin = 'topicin'

kafkaTargetArgs = Args()
kafkaTargetArgs.ip = '192.168.1.144'
kafkaTargetArgs.port = '9092'
kafkaTargetArgs.topicout = 'topicout'

nnResNet50v2Args = Args();
nnResNet50v2Args.name = 'ResNet50args'

nnResNet50v2 = NNResNet50v2(nnResNet50v2Args)
#nnResNet50v2.execute(KafkaSource(kafkaSourceArgs), TargetTemplate(targetArgs))
nnResNet50v2.execute(KafkaSource(kafkaSourceArgs), KafkaTarget(kafkaTargetArgs))

del nnResNet50v2