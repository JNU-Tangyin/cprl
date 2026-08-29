from .Autoformer import Model as Autoformer
from .FEDformer import Model as FEDformer
from .iTransformer import Model as iTransformer
# from .Mamba import Model as Mamba
from .Informer import Model as Informer
from .ETSformer import Model as ETSformer
from .TimesNet import Model as TimesNet
from .Reformer import Model as Reformer
from .SCINet import Model as SCINet
from .TemporalFusionTransformer import Model as TemporalFusionTransformer
from .Pyraformer import Model as Pyraformer
from .Nonstationary_Transformer import Model as Nonstationary_Transformer
from .PAttn import Model as PAttn
from .Transformer import Model as Transformer
from .TSMixer import Model as TSMixer
from .TimeXer import Model as TimeXer

# MODEL_REGISTRY 用来通过字符串名字找到对应模型类
MODEL_REGISTRY = {
    "Autoformer": Autoformer,
    "FEDformer": FEDformer,
    "iTransformer": iTransformer,
    # "Mamba": Mamba,
    "Informer": Informer,
    "ETSformer": ETSformer,
    "TimesNet": TimesNet,
    "Reformer": Reformer,
    "SCINet": SCINet,
    "TemporalFusionTransformer": TemporalFusionTransformer,
    "Pyraformer": Pyraformer,
    "Nonstationary_Transformer": Nonstationary_Transformer,
    "PAttn": PAttn,
    "Transformer": Transformer,
    "TSMixer": TSMixer,
    "TimeXer": TimeXer,
}
