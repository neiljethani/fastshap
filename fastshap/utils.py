import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

import numpy as np

#### Links #### (adapted from https://github.com/slundberg/shap/blob/master/shap/utils/_legacy.py)

class Link:
    def __init__(self):
        pass
    
# Numpy
class IdentityLink(Link):
    def __str__(self):
        return "identity"

    @staticmethod
    def f(x):
        return x

    @staticmethod
    def finv(x):
        return x

class LogitLink(Link):
    def __str__(self):
        return "logit"

    @staticmethod
    def f(x):
        return np.log(x/(1-x))

    @staticmethod
    def finv(x):
        return 1/(1+np.exp(-x))


def convert_to_link(val):
    if isinstance(val, Link):
        return val
    elif val == "identity":
        return IdentityLink()
    elif val == "logit":
        return LogitLink()
    else:
        assert False, "Passed link object must be a subclass of iml.Link"


# Tensorflow
class IdentityLinkTF(Link):
    def __init__(self):
        pass
    
    def __str__(self):
        return "identity"

    @staticmethod
    def f(x):
        return x

    @staticmethod
    def finv(x):
        return x


class LogitLinkTF(Link):
    def __init__(self):
        pass
    
    def __str__(self):
        return "logit"

    @staticmethod
    def f(x):
        return tf.math.log(x/(1-x))

    @staticmethod
    def finv(x):
        return 1/(1+tf.math.exp(-x))


def convert_to_linkTF(val):
    if isinstance(val, Link):
        return val
    elif val == "identity":
        return IdentityLinkTF()
    elif val == "logit":
        return LogitLinkTF()
    else:
        assert False, "Passed link object must be a subclass of iml.Link"
        
        
#### Shapley Sampler ####

class ShapleySampler(Layer):
    '''
    Layer to Sample S according to the Shapley Kernel Weights
    '''
    def __init__(self, num_features, paired_sampling=True, num_samples=1, **kwargs):
        super(ShapleySampler, self).__init__(**kwargs)
        
        self.num_features = num_features
        
        # Weighting kernel (probability of each subset size). 
        #credit = https://github.com/iancovert/sage/blob/master/sage/kernel_estimator.py
        w = tf.range(1, num_features)
        w = 1 / (w * (num_features - w))
        self.w = w / K.sum(w)
        
        self.paired_sampling = paired_sampling
        self.num_samples = num_samples
        
        self.ones_matrix = tf.linalg.band_part(
            tf.ones((num_features,num_features), tf.int32), 
            -1, 0)
    
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        
        # Sample subset size = number of features to select in each sample
        num_included = tf.random.categorical(
            tf.expand_dims(tf.math.log(self.w), 0), batch_size * self.num_samples
        )
        num_included = tf.transpose(num_included, [1,0])
        
        S = tf.gather_nd(self.ones_matrix, num_included)
        S = tf.map_fn(tf.random.shuffle, S)
        
        # Uniformly sample features of subset size
        S = tf.reshape(S, [batch_size, self.num_samples, self.num_features])
        
        #Paried Sampling 
        if self.paired_sampling:
            S_complement = 1 - S
            S = tf.concat([S, S_complement], axis = 1)
        
        return S
    
    def get_config(self):
        config = super(ShapleySampler, self).get_config()
        config.update({"num_features": self.num_features})
        config.update({"paired_sampling": self.paired_sampling})
        config.update({"num_samples": self.num_samples})
        config.update({"ones_matrix": self.ones_matrix})
        config.update({"w": self.w})
        return config