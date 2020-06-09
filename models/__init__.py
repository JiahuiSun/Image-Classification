
from .vgg import *

__factory = {

    'vgg': VGGNet,
}


def get_names():
    return __factory.keys()


def get_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
