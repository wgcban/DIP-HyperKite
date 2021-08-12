from .HyperPNN import *
from .DHP_DARN import *
from .Spectral_Spatial_GR import *
from .HPF import *
from .kiunet import *
from .HyperTransformer import *

MODELS = {  "HyperPNN": HyperPNN, 
            "DHP_DARN": DHP_DARN,
            "DHP_SSGR": DHP_SSGR,
            "DHP_SSGRV2": DHP_SSGRV2,
            "DHP_SSGRV3": DHP_SSGRV3,
            "HPF": HPF,
            "RDN": RDN,
            "kitenet": kitenet,
            "kitenetwithsk": kitenetwithsk,
            "kiunet": kiunet,
            "attentionkitenet": attentionkitenet,
            "HSIT": HyperTransformer,
            "HSIT_PRE": HyperTransformerPre}