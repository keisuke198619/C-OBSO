from .rnn_gauss import RNN_GAUSS
from .macro_vrnn import MACRO_VRNN
from .gvrnn import GVRNN

def load_model(model_name, params, parser=None):
    model_name = model_name.lower()

    if model_name == 'rnn_gauss':
        return RNN_GAUSS(params, parser)
    elif model_name == 'macro_vrnn':
        return MACRO_VRNN(params, parser)
    elif model_name == 'gvrnn':
        return GVRNN(params, parser)
    else:
        raise NotImplementedError
