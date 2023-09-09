#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import Library
import numpy as np
import pandas as pd
get_ipython().system('pip install keras_tuner')
import keras_tuner

import keras
from keras import models
from keras import layers
from keras.layers import Dense, BatchNormalization, Dropout
from keras import metrics
import tensorflow as tf
from keras.regularizers import l1
from sklearn.metrics import mean_squared_error

import tensorflow as tf
import random
from keras.layers import Dense, Dropout, BatchNormalization
from keras.regularizers import l1_l2
from keras import layers, Sequential
from kerastuner import HyperModel

import kerastuner as kt


# In[ ]:


# Import Dataset
path = '/kaggle/input/paper1/'
firm_df = pd.read_pickle(path+'firm_df49.pkl')
macro_df = pd.read_pickle(path+'macro_df.pkl')
macro_df.index = pd.to_datetime(macro_df.index, format='%m/%d/%Y')
firm_df.index = pd.to_datetime(firm_df.index).to_period('M')


# In[ ]:


# Split Training, Validation and test

class DataPrepare():
    def __init__(self):
        self.char_core = ['acc','agr','beta','bm','cash','cashpr','cfp','chatoia','chcsho','chfeps','chinv',
                          'chmom','chpmia','chtx','currat','depr','dy','ear','ep','gma','grcapx','grltnoa',
                          'ill','indmom','invest','lev','lgr','maxret','mom12m','mom1m','mom36m','mve','nincr',
                          'orgcap','pchgm_pchsale','pchsale_pchinvt','pchsale_pchrect','pchsale_pchxsga',
                          'retvol','roaq','roavol','roeq','salecash','saleinv','sgr','sp','std_dolvol','std_turn',
                          'turn']
        self.info_list = ['fyear','year','jyear','permno','ticker','comnam','exchcd','exchname','siccd',
                          'indname','size_class','mve_m','rf','ret','ret_adj','ret_ex','ret_adj_ex',]

    def datasplit(self, df):
        df_train = df[df.index < pd.Period((str(1995)+"-1"),freq='M')]
        df_val = df[(df.index >= pd.Period((str(1995)+"-1"),freq='M')) & (df.index < pd.Period((str(2000)+"-1"),freq='M'))]
        df_test = df[df.index >= pd.Period((str(2000)+"-1"),freq='M')]
        X_train = df_train[self.char_core]
        y_train = df_train['predicted_return']
        X_val = df_val[self.char_core]
        y_val = df_val['predicted_return']
        X_test = df_test[self.char_core]
        y_test = df_test['predicted_return']
        return (X_train, y_train, X_val, y_val, X_test, y_test, )

prepare = DataPrepare()
prepare = DataPrepare()
X_train, y_train, X_val, y_val, X_test, y_test = prepare.datasplit(firm_df)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)


# In[ ]:


# HyperParameter Tuning
class MyHyperModel(HyperModel):
    def __init__(self, input_shape, random_seed=None):
        self.input_value = input_shape
        self.acti = 'relu'
        self.epoch = 200
        self.patience = 3

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)

    def my_metric_fn(self, y_true, y_pred):
        num = tf.reduce_mean(tf.square(y_true - y_pred))
        den = tf.reduce_mean(tf.square(y_true - tf.zeros_like(y_true)))
        return 1 - num / den

    def call_existing_code(self, l1_reg, l2_reg, dropout, L, lr):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.input_value, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg), activation=self.acti))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        for i in range(L - 1):
            model.add(Dense(int(32/2**i), kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg), activation=self.acti))
            model.add(Dropout(dropout))
            model.add(BatchNormalization())
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[self.my_metric_fn])
        return model

    def build(self, hp, *args, **kwargs):
        l1_reg = hp.Choice("l1_ratio", [0.,])
        l2_reg = hp.Choice("l2_ratio", [0.1, ])
        layer = hp.Choice("layer", [4,])
        dropout = hp.Choice("dropout", [0.,])
        lr = hp.Choice("learning_rate", [0.0001])
        model = self.call_existing_code(
            l1_reg=l1_reg, l2_reg=l2_reg, L=layer, dropout=dropout, lr=lr
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
        *args,
        batch_size=hp.Choice("batch_size", [128,]),
        epochs=self.epoch,
        **kwargs,
    )

# Set random seed for reproducibility during tuning
import kerastuner as kt

random_seed = 1120
tuner = kt.RandomSearch(
    MyHyperModel(input_shape=49, random_seed=random_seed),
    objective="val_loss",
    max_trials=100,
    directory="my_dir",
    overwrite=True,
    project_name="tune_hypermodel",
)
tuner.search(X_train, y_train, validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
            )


# In[ ]:


# 10 Trials for Prediction 
x_all = np.concatenate((X_train, X_val))
y_all = np.concatenate((y_train, y_val))

best_hps = tuner.get_best_hyperparameters(5)
print(best_hps[0].values)


seed_list = [458, 165, 530, 564, 590, 560, 829, 170, 376, 176]
yhat_df = pd.DataFrame()

for random_seed in seed_list:
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    best_model = tuner.get_best_models(3)[0]
    best_model.fit(x=x_all, y=y_all, epochs=1,
              callbacks=[keras.callbacks.EarlyStopping(monitor='my_metric_fn',mode='max', patience=1)])
    print(best_model.evaluate(X_test,y_test))
    y_hat = best_model.predict(X_test).reshape(-1)
    yhat_df[random_seed] = y_hat
    print()
    
yhat_df.to_pickle('yhat_NN1.pkl')


# In[ ]:


# Print out Predictive R^2
y_predict = yhat_df.mean(axis=1).values.reshape(-1)
y_real = y_test

a = np.mean(np.square(y_predict -  y_real))
b = np.mean(np.square(y_real))
print(1-a/b)

