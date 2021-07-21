import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense, Lambda, Reshape, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from datetime import datetime
import os

from fastshap.utils import ShapleySampler


class Surrogate:
    
    '''
    Wrapper around surrogate model.
    Args:
      surrogate:
    '''
    
    def __init__(self, 
                 surrogate_model, 
                 num_features, 
                 baseline):
        
        # Models
        self.surrogate_model = surrogate_model
        
        # Parameters
        self.P = num_features
        self.baseline = baseline
        
        
        
              
        
    def train(self,
              original_model,
              train_data,
              val_data,
              batch_size,
              max_epochs,
              validation_batch_size,
              loss_fn="categorical_crossentropy",
              lr=1e-3,
              min_lr=1e-5,
              lr_factor=0.9,
              lookback=10,
              verbose=False, 
              model_dir = None):
        
        # model save dir
        if model_dir is None:
            self.save = datetime.now().strftime("%Y%m%d_%H_%M_%S")
            self.model_dir = os.path.join(os.getcwd(), self.save)
        else:
            self.model_dir = model_dir
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
            
        # Data
        #Check if Provided TF Dataset, if So X should be paired with model predictions
        if (isinstance(train_data, tf.python.data.ops.dataset_ops.PrefetchDataset)              
            or isinstance(train_data, tf.python.data.ops.dataset_ops.MapDataset)): 
            @tf.function
            def make_prediction_data(x, y):
                with tf.device(device):
                    y_model = original_model(x)
                return (x, y_model)

            with tf.device(device):
                train_data = train_data.map(make_prediction_data)
                val_data = val_data.map(make_prediction_data)
            
        else:
            fx_train = original_model.predict(train_data)
            fx_val = original_model.predict(val_data)
            train_data = tf.data.Dataset.from_tensor_slices((train_data, fx_train)).batch(batch_size)
            val_data = tf.data.Dataset.from_tensor_slices((val_data, fx_val)).batch(validation_batch_size)
        
        #################################################################
        
        #Make Model w/ Masking
        self.surrogate_model.trainable = True
        
        model_input = Input(shape=self.P, dtype='float32', name='input')
        S = ShapleySampler(self.P, paired_sampling=False, num_samples=1)(model_input)
        S = Lambda(lambda x: tf.cast(x, tf.float32))(S)
        S = Reshape((self.P,))(S)
        x_S = Lambda(lambda x: x[1]*x[0] + (1-x[1])*self.baseline)([model_input, S])
        
        out = self.surrogate_model(x_S)
        
        self.model = Model(model_input, out)
        
        # Metrics
        METRICS = [ 
          tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='accuracy'),
        ]
        
        # Model Checkpointing
        weights_path = os.path.join(self.model_dir, 'surrogate_weights.h5')
        checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, 
                                     save_best_only=True, mode='min', save_weights_only = True)
        
        # LR Schedule
        reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=2, 
                                     verbose=1, mode='min', cooldown=1, min_lr=min_lr)
        
        # Early Stopping 
        earlyStop = EarlyStopping(monitor="val_loss", mode="min", patience=lookback) 
        
        # Compile Model
        CALLBACKS = [checkpoint, earlyStop, reduceLR]
        OPTIMIZER = tf.keras.optimizers.Adam(lr)

        self.model.compile(
            loss=loss_fn,
            optimizer=OPTIMIZER,
            metrics=METRICS,
        )
        
        # Train Model 
        self.model.fit(x = train_data,
                       epochs = max_epochs,
                       validation_data = val_data,
                       callbacks = CALLBACKS, 
                       verbose=verbose)
        
        
        # Get Checkpointed Model
        self.model.load_weights(weights_path)
        
        # Remove Masking Layer
        self.model = self.model.layers[-1]
        self.model.trainable = False
        # Save Model
        self.model.save(os.path.join(self.model_dir, 'surrogate.h5'))
        

class ImageSurrogate:
    
    '''
    Wrapper around surrogate model.
    Args:
      surrogate_model:
      baseline:
      width:
      height:
      superpixel_size:
      model_dir:
    '''
    
    def __init__(self, 
                 surrogate_model,
                 baseline,
                 width, height, 
                 superpixel_size):
        
        # Models
        self.surrogate_model = surrogate_model
        
        # Verify arguments.
        assert width % superpixel_size == 0
        assert height % superpixel_size == 0

        # Set up superpixel upsampling.
        self.width = width
        self.height = height
        self.superpixel_size = superpixel_size
        if superpixel_size == 1:
            self.upsample = Lambda(lambda x: x)
        else:
            self.upsample = UpSampling2D(
                size=(superpixel_size, superpixel_size))
            
        # Set up number of players.
        self.small_width = width // superpixel_size
        self.small_height = height // superpixel_size
        self.P = self.small_width * self.small_height
        
        # Set baseline masking value
        self.baseline = baseline
              
        
    def train(self,
              original_model,
              train_data,
              val_data,
              batch_size,
              max_epochs,
              validation_batch_size,
              loss_fn='categorical_crossentropy',
              lr=1e-3,
              min_lr=1e-5,
              lr_factor=0.9,
              lookback=10,
              gpu_device=0,
              verbose=False, 
              model_dir=None):
        
        # model save dir
        if model_dir is None:
            self.save = datetime.now().strftime("%Y%m%d_%H_%M_%S")
            self.model_dir = os.path.join(os.getcwd(), self.save)
        else:
            self.model_dir = model_dir
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Device
        if gpu_device is False:
            device = "cpu:0"
        else:
            device = "gpu:" + str(gpu_device)
        
        # Data
        #Check if Provided TF Dataset, if So X should be paired with model predictions
        if (isinstance(train_data, tf.python.data.ops.dataset_ops.PrefetchDataset)              
            or isinstance(train_data, tf.python.data.ops.dataset_ops.MapDataset)): 
            @tf.function
            def make_prediction_data(x, y):
                with tf.device(device):
                    y_model = original_model(x)
                return (x, y_model)

            with tf.device(device):
                train_data = train_data.map(make_prediction_data)
                val_data = val_data.map(make_prediction_data)
            
        else:
            fx_train = original_model.predict(train_data)
            fx_val = original_model.predict(val_data)
            train_data = tf.data.Dataset.from_tensor_slices((train_data, fx_train))
            val_data = tf.data.Dataset.from_tensor_slices((val_data, fx_val))
        
        #################################################################
        
        #Make Model w/ Masking
        self.surrogate_model.trainable = True
        
        model_input = Input(shape=original_model.input.shape[1:], dtype='float32', name='input')
        S = ShapleySampler(self.P, paired_sampling=False, num_samples=1)(model_input)
        S = Lambda(lambda x: tf.cast(x, tf.float32))(S)
        S = Reshape((self.small_height,self.small_width,1))(S)
        S = self.upsample(S)
        x_S = Lambda(lambda x: x[1]*x[0] + (1-x[1])*self.baseline)([model_input, S])
        
        out = self.surrogate_model(x_S)
        
        self.model = Model(model_input, out)
        
        # Metrics
        METRICS = [ 
          tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='accuracy'),
        ]
        
        # Model Checkpointing
        weights_path = os.path.join(self.model_dir, 'surrogate_weights.h5')
        checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=verbose, 
                                     save_best_only=True, mode='min', save_weights_only = True)
        
        # LR Schedule
        reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=2, 
                                     verbose=1, mode='min', cooldown=1, min_lr=min_lr)
        
        # Early Stopping 
        earlyStop = EarlyStopping(monitor="val_loss", mode="min", patience=lookback) 
        
        # Compile Model
        CALLBACKS = [checkpoint, earlyStop, reduceLR]
        OPTIMIZER = tf.keras.optimizers.Adam(lr)

        self.model.compile(
            loss=loss_fn,
            optimizer=OPTIMIZER,
            metrics=METRICS,
        )
        
        # Train Model 
        self.model.fit(x = train_data,
                       epochs = max_epochs,
                       validation_data = val_data,
                       validation_batch_size = validation_batch_size,
                       callbacks = CALLBACKS,
                       verbose=verbose)
        
        
        # Get Checkpointed Model
        self.model.load_weights(weights_path)
        
        # Remove Masking Layer
        self.model = self.model.layers[-1]
        self.model.trainable = False
        # Save Model
        self.model.save(os.path.join(self.model_dir, 'surrogate.h5'))
        