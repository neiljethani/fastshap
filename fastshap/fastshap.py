import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense, Lambda, Reshape, TimeDistributed, Permute, UpSampling3D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from datetime import datetime
import os

from fastshap.utils import convert_to_linkTF, convert_to_link, ShapleySampler

class EfficiencyNormalization(Layer):
    '''
    Layer to peform efficiency normalization.
    '''
    def __init__(self, normalization, null, link, P, **kwargs):
        super(EfficiencyNormalization, self).__init__(**kwargs)
        
        self.null = null
        self.link = link
        self.P = P
        self.normalization = normalization
        
        
    def call(self, inputs):
        phi = inputs[0]
        grand = self.link(K.stop_gradient(K.clip(inputs[1], 1e-7, 1-1e-7)))
        null = self.link(tf.constant(self.null, dtype=tf.float32))
         
        # Calculate gap for Efficiency regularization/ Additive Normalization
        gap = grand-null - K.sum(phi, -1)
        
        #Normalize
        if self.normalization == 'additive':
            phi += tf.expand_dims(gap/self.P, -1)
        elif self.normalization == 'multiplicative':
            ratio = grand-null/K.sum(phi, -1)
            phi *= tf.expand_dims(ratio, -1)
            
        return phi, gap
            
    def get_config(self):
        config = super(EfficiencyNormalization, self).get_config()
        config.update({"null": self.null})
        config.update({"link": self.link})
        config.update({"P": self.P})
        config.update({"normalization": self.normalization})
        return config
        
        

class FastSHAP:
    '''
    Wrapper around FastSHAP explanation model.

    Args:
      explainer:
      imputer:
      normalization:
      link:
      baseline:
    '''

    def __init__(self,
                 explainer,
                 imputer,
                 normalization=None,
                 link=None,
                 baseline = 0):
        
        # Set up explainer, imputer and link function.
        self.explainer = explainer
        self.imputer = imputer
        
        # Parameters
        self.P = imputer.P
        self.D = imputer.surrogate_model.output.shape[-1]
        self.input_shape = explainer.input.shape[1:]
        
        #Link
        self.linkfv =  np.vectorize(convert_to_link(link).f)
        self.link = convert_to_linkTF(link)
        
        #Baseline for Masking 
        self.baseline = baseline
        
        # Null
        self.null = np.squeeze(imputer.surrogate_model.predict(np.zeros(tuple([1]+list(self.input_shape)))))
        
        # Set up normalization.
        if normalization is None or normalization in ['additive', 'multiplicative']:
            self.normalization = normalization
        else:
            raise ValueError('unsupported normalization: {}'.format(
                normalization))
            
            
    def train(self,
              train_data,
              val_data,
              batch_size,
              num_samples,
              max_epochs,
              validation_batch_size,
              lr=2e-4,
              min_lr=1e-5,
              lr_factor=0.5,
              eff_lambda=0,
              paired_sampling=True,
              lookback=5,
              verbose=False,  
              model_dir = None):
        '''
        Train explainer model.

        Args:
          train_data:
          val_data:
          batch_size:
          num_samples:
          max_epochs:
          lr:
          min_lr:
          lr_factor:
          eff_lambda:
          paired_sampling:
          lookback:
          verbose:
          model_dir:
        '''
        
        # model save dir
        if model_dir is None:
            self.save = datetime.now().strftime("%Y%m%d_%H_%M_%S")
            self.model_dir = os.path.join(os.getcwd(), self.save)
        else:
            self.model_dir = model_dir
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Data (get dummy labels)
        if not (isinstance(train_data,tf.python.data.ops.dataset_ops.PrefetchDataset) or isinstance(train_data, tf.python.data.ops.dataset_ops.MapDataset)): 
            y_train_dummy = np.zeros((train_data.shape[0],1))
            y_val_dummy = np.zeros((val_data.shape[0],1))
            train_data = tf.data.Dataset.from_tensor_slices((train_data, y_train_dummy)).batch(batch_size)
            val_data = tf.data.Dataset.from_tensor_slices((val_data, y_val_dummy)).batch(validation_batch_size)
            
            
        
        ###### Create Model ######
        model_input = Input(shape=self.input_shape, dtype='float32', name='input')
        S = ShapleySampler(self.P, paired_sampling=paired_sampling, num_samples = num_samples)(model_input)
        S = Lambda(lambda x: tf.cast(x, tf.float32), name='S')(S)
        
        #If Paired Double num_samples:
        if paired_sampling:
            num_samples = 2 * num_samples
        else:
            num_samples = num_samples
              
        # Learn Phi 
        phi = self.explainer(model_input)
        if len(self.input_shape) == 3:
            # Image.
            image_shape = self.input_shape
            phi = Reshape((self.P, self.D))(phi)
            phi = Permute((2,1))(phi)
        else:
            # Tabular.
            image_shape = None
            phi = Reshape((self.D, self.P))(phi)
        
        #Efficency Normalization
        phi = Layer(name='phi')(phi)
        grand = self.imputer.surrogate_model(model_input)
        phi, gap = EfficiencyNormalization(self.normalization, self.null, 
                                           self.link.f, self.P)([phi, grand])
            
        # Name Output Layer and Reshape
        phi = Reshape((self.P*self.D,))(phi)
            
        # Repeat Phi for Multiple Subset Sampling
        phi_repeat = tf.keras.layers.RepeatVector(num_samples)(phi)
        phi_repeat = Reshape((num_samples, self.D, self.P),  name='phi_repeat')(phi_repeat)
        
        # Calculate output 
        phi_S = Lambda(lambda x: tf.concat([x[0], tf.expand_dims(x[1], 2)], 2))([phi_repeat, S])
        out = TimeDistributed(
            Lambda(lambda x: 
                   tf.squeeze(tf.matmul(x[:,:self.D,:], tf.expand_dims(x[:,-1,:], -1)), -1)),
            name = 'linear_model'
        )(phi_S)
        
        # Repeat Input for Multiple Subset Sampling
        if image_shape is not None:
            #Image
            model_input_repeat = Reshape(tuple([1]+list(image_shape)))(model_input)
            model_input_repeat = UpSampling3D(size=(num_samples, 1, 1),
                                              name='model_input_repeat')(model_input_repeat)
            
            # Resize Masks
            S = TimeDistributed(Reshape((self.imputer.small_height, self.imputer.small_width, 1)))(S)
            S = TimeDistributed(self.imputer.upsample, name='S_RM')(S)
        else:
            #Tabular
            model_input_repeat = Reshape((1, self.P))(model_input)
            model_input_repeat = tf.keras.layers.UpSampling1D(size=num_samples, name='model_input_repeat')(model_input_repeat)
            
            
        # yAdj = link(f(x_s))- link(E[f(x)])
        xs = Lambda(lambda x: x[1]*x[0] + (1-x[1])*self.baseline)([model_input_repeat, S])
        f_xs = TimeDistributed(self.imputer.surrogate_model, name='f_xs')(xs)
        yAdj = TimeDistributed(
            Lambda(lambda x: K.stop_gradient(
                self.link.f(K.clip(x, 1e-7, 1-1e-7)) - self.link.f(tf.constant(self.null, dtype=tf.float32))
            )), name = 'yAdj'
        )(f_xs)
        
        ## MSE Loss w/ Efficiency Regularization         
        SHAPloss = tf.reduce_mean(tf.keras.losses.MSE(yAdj, out))
        EFFloss = eff_lambda*tf.reduce_mean(gap**2) 
 
        self.explainer = Model(model_input, out)
                                         
        self.explainer.add_loss(SHAPloss)
        self.explainer.add_loss(EFFloss)
        
        self.explainer.add_metric(SHAPloss, name='shap_loss', aggregation='mean')
        self.explainer.add_metric(EFFloss, name='eff_loss', aggregation='mean')
        
        # Model Checkpointing
        explainer_weights_path = os.path.join(self.model_dir, 'explainer_weights.h5')
        checkpoint = ModelCheckpoint(explainer_weights_path, monitor='val_shap_loss', verbose=verbose, 
                                     save_best_only=True, mode='min', save_weights_only = True)
        
        # Early Stopping 
        earlyStop = EarlyStopping(monitor="val_shap_loss", mode="min", patience=lookback) 
        
        # LR Schedule
        reduceLR = ReduceLROnPlateau(monitor='val_shap_loss', factor=lr_factor, patience=2, 
                                     verbose=1, mode='min', cooldown=1, min_lr=min_lr)
        
        # Compile Model
        CALLBACKS = [checkpoint, earlyStop, reduceLR]
        OPTIMIZER = tf.keras.optimizers.Adam(lr)
        
        self.explainer.compile(
            optimizer=OPTIMIZER
        )
        
        # Train Model
        history = self.explainer.fit(x = train_data,
                                     epochs = max_epochs,
                                     validation_data = val_data,
                                     validation_batch_size = validation_batch_size,
                                     callbacks = CALLBACKS, 
                                     verbose=verbose)
        
        self.val_losses = history.history['val_shap_loss']
        
        # Get Checkpointed Model
        self.explainer.load_weights(explainer_weights_path)
        
        #  Extract Explainer
        # 1) Get Base Model for Phi
        base_model = Model(self.explainer.get_layer('input').input, 
                           self.explainer.get_layer('phi').output)
        base_model.summary()
        
        # 2) Resize
        model_input = Input(shape=self.input_shape, dtype='float32')
        phi = base_model(model_input)
        
        if image_shape is not None:
            #Image
            phi = Permute((2,1))(phi)
            phi = Reshape((self.imputer.small_height, self.imputer.small_width, self.D))(phi)
            phi = self.imputer.upsample(phi)
            phi = Rescaling(1./self.imputer.superpixel_size, offset=0.0)(phi)

        self.explainer = Model(model_input, phi)
        self.explainer.trainable = False
        # Save Model
        self.explainer.save(os.path.join(self.model_dir, 'explainer.h5'))
        
    def shap_values(self, X):
        explanations = self.explainer.predict(X)
            
        # efficeny normalization
        if self.normalization == 'additive':
            prediction = self.linkfv(self.imputer.surrogate_model.predict(X)) - self.linkfv(self.null)
            diff = (prediction - explanations.sum(-1))
            explanations += np.expand_dims(diff/explanations.shape[-1], -1)
        elif self.normalization == 'multiplicative':
            prediction = self.linkfv(self.imputer.surrogate_model.predict(X)) - self.linkfv(self.null)
            ratio = (prediction/explanations.sum(-1))
            explanations *= np.expand_dims(ratio, -1)
            
        return explanations
        
            
        