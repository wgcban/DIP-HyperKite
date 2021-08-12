from .HyperPNN import *
from .DHP_DARN import *
from .Spectral_Spatial_GR import *
from .HPF import *
from .kiunet import *
from .HyperTransformer import *

MODELS = {  "HyperPNN": HyperPNN, 
            "DHP_DARN": DHP_DARN,
            "HPF": HPF,
            "RDN": RDN,
            "kitenet": kitenet,
            "kitenetwithsk": kitenetwithsk,
            "kiunet": kiunet,
            "attentionkitenet": attentionkitenet}