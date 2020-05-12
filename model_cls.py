from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Input, BatchNormalization, Dense
from keras.layers import Reshape, Lambda, concatenate
from keras.models import Model
from keras.engine.topology import Layer
from tensorflow.python.keras import backend as K
import numpy as np
import tensorflow as tf
import globals as _g

def _mlp1(input_shape):
    inputs = Input(shape=input_shape, name='inputs')
    # forward net
    g = Conv1D(64, 1, activation='relu')(inputs)
    g = BatchNormalization()(g)
    g = Conv1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)

    # forward net
    g = Conv1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(128, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(1024, 1, activation='relu')(g)
    g = BatchNormalization()(g)

    # global feature
    global_feature = MaxPooling1D(pool_size=1024)(g)
    mlp=Model(inputs=inputs, outputs=global_feature, name='mlp1')
    return mlp

def _split_inputs(inputs):
    """
    split inputs to NUM_VIEW input
    :param inputs: a Input with shape VIEWS_IMAGE_SHAPE
    :return: a list of inputs which shape is IMAGE_SHAPE
    """
    slices = []
    for i in range(0, _g.NUM_VIEWS):
        slices.append(inputs[:,i*1024:(i+1)*1024,:])
    return slices

def _view_pool(views):
    expanded = [K.expand_dims(view, 0) for view in views]
    concated = K.concatenate(expanded, 0)
    reduced = K.max(concated, 0)
    #reduced = MaxPooling1D(pool_size=1024)(views)
    return reduced

def MVPointNet(nb_classes):
    inputs = Input(shape=(_g.NUM_VIEWS*1024,3),name='input')
    views = Lambda(_split_inputs, name='split')(inputs)

    mlp1_model = _mlp1(_g.PC_SHAPE)
    if _g.NUM_VIEWS==1:
        pool=mlp1_model(views)
    else:
        view_pool=[]
        for view in views:
            view_pool.append(mlp1_model(view))

        pool=Lambda(_view_pool, name='view_pool')(view_pool)


    # point_net_cls
    c = Dense(512, activation='relu')(pool)
    c = BatchNormalization()(c)
    c = Dropout(0.5)(c)
    c = Dense(256, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(0.5)(c)
    c = Dense(nb_classes, activation='softmax')(c)
    prediction = Flatten()(c)

    model = Model(inputs=inputs, outputs=prediction)

    return model
