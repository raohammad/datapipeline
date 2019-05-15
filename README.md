# nndatapipeline
NN Data Pipeline for Inferencing on Neural Networks (onnx fundamentally). 

Designed roughly on pipeline design pattern. The NN is connected to source and target that implement abstrat functions of source.Base and target.Base classes respectively. New NNs can be added that need to be inherited from nn.Base class and consequently implementing all abstract functions. This repository is a part of larger implementation but is a cohesive module in itself. Its made opensource for community contribution to add as many source and targets as possible so that NN pipelines can be built with as many adapters support as possible.

# source
Source adapters can be added to support data ingestion from multipe sources. Each source adapter needs to be extended from source.Base class and implements, `delegate(..)` function

`delegate` function accepts callback function that is passed from NN class derived from `nn.Base`

`SourceTemplate` class is a sample implementation that dumps data received on screen

# target
Source adapters can be added to support data ingestion from multipe sources. Each source adapter needs to be extended from target.Base class and implement three functions, `dumpData(..)` function

`dumpData` function accepts data that needs to be saved/forwarded as a result of prediction on `nn.Base`

`TargetTemplate` class is a sample implementation that dumps data received on screen

# nn
NNs that needs to be added to enhance functions of the framework need to extend from base class `nn.Base`. Every NN consequently implements four abstract methods;

`execute` is the entry point to the NN utilized by the driver program it accepts the primary source to receive from and primary target to dumpData to

`callback` is the entry point to the NN utilized by the *source* with data received. It calls __self__ functions in below order. Afterwards it calls `dumpData` function of target with prediction results

`preprocess` if the data has already passed through ETL pipeline, its anticipate that the data is already prepared for this NN utilization. Nethertheless there could be instances when data needs preprocessing before passing on for prediction. This function serves that purpose

`predict` preprocessed data is then passed on to this method where the NN model is invoked with processed data

