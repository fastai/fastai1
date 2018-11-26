import pytest
from fastai import *
from fastai.vision import *

def _print_data(data): print(len(data.train_ds),len(data.valid_ds))
def _check_data(data, t, v):
    assert len(data.train_ds)==t
    assert len(data.valid_ds)==v
    _ = data.train_ds[0]

def test_image_to_image_different_tfms():
    get_y_func = lambda o:o
    mnist = untar_data(URLs.COCO_TINY)
    x_tfms = get_transforms()
    y_tfms = [[t for t in x_tfms[0]], [t for t in x_tfms[1]]]
    y_tfms[0].append(flip_lr())
    data = (ImageItemList.from_folder(mnist)
            .random_split_by_pct()
            .label_from_func(get_y_func)
            .transform(x_tfms)
            .transform_y(y_tfms)
            .databunch(bs=16))

    x,y = data.one_batch()
    x1 = x[0]
    y1 = y[0]
    x1r = flip_lr(Image(x1)).data
    assert (y1 == x1r).all()
