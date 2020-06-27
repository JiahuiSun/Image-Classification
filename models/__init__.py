
from .vgg import *

__factory = {
    'dnn': DNN,
    'vgg': VGGNet,
    'alex': AlexNet,
}


def get_names():
    return __factory.keys()


def get_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
