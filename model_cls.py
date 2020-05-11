from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Input, BatchNormalization, Dense
from keras.layers import Reshape, Lambda, concatenate
from keras.models import Model
from keras.engine.topology import Layer
from tensorflow.python.keras import backend as K
import numpy as np
import tensorflow as tf

def view_pool(views):
    expanded = [K.expand_dims(view, 0) for view in views]
    concated = K.concatenate(expanded, 0)
    reduced = K.max(concated, 0)
    return reduced

def MVPointNet(nb_classes,num_view):
    input_points = Input(shape=(num_view*1024, 3))

    # forward net
    g = Conv1D(64, 1, activation='relu')(input_points)
    g = BatchNormalization()(g)
    g = Conv1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)


    #g = view_pool(g)

    # forward net
    g = Conv1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(128, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Conv1D(1024, 1, activation='relu')(g)
    g = BatchNormalization()(g)

    # global feature
    global_feature = MaxPooling1D(pool_size=1024)(g)

    # point_net_cls
    c = Dense(512, activation='relu')(global_feature)
    c = BatchNormalization()(c)
    c = Dropout(0.5)(c)
    c = Dense(256, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(0.5)(c)
    c = Dense(nb_classes, activation='softmax')(c)
    prediction = Flatten()(c)

    model = Model(inputs=input_points, outputs=prediction)

    return model
