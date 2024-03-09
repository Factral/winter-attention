# Inspired by
# https://github.com/rwightman/pytorch-image-models/blob/b9bd960a032c75ca6b808ddeed76bee5f3ed4972/timm/models/layers/helpers.py
import collections.abc

def to_2tuple(x):
    if isinstance(x, collections.abc.iterable):
        return x
    return (x,x)
