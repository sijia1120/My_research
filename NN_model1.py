from keras_tuner import HyperModel
import keras_tuner as kt

import numpy as np
import pandas as pd
#!pip install keras_tuner
#import keras_tuner

import keras
from keras import models, layers, metrics, Sequential
from keras.layers import Dense, BatchNormalization, Dropout
import tensorflow as tf
from keras.regularizers import l1,l2,l1_l2
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Input, Dense, Multiply, Concatenate
from tensorflow.keras.models import Model

import tensorflow as tf
import random

import keras_tuner
from keras_tuner import HyperModel
import keras_tuner as kt

% ------------------------
char_core = ['acc','agr','beta','bm','cash','cashpr','cfp','chatoia','chcsho','chfeps','chinv',
             'chmom','chpmia','chtx','currat','depr','dy','ear','ep','gma','grcapx','grltnoa',
             'ill','indmom','invest','lev','lgr','maxret','mom12m','mom1m','mom36m','mve','nincr',
             'orgcap','pchgm_pchsale','pchsale_pchinvt','pchsale_pchrect','pchsale_pchxsga',
             'retvol','roaq','roavol','roeq','salecash','saleinv','sgr','sp','std_dolvol','std_turn','turn']

#merged_latent.to_pickle('merged_latent49.pkl')
merged_latent49 = pd.read_pickle("merged_latent49.pkl")

firm_col = merged_latent49.columns.tolist()[18:18+49]
latent_col = merged_latent49.columns.tolist()[18+49+1:]
print(len(firm_col), len(latent_col))

% ------------------------------------------------------------------------
df_train = merged_latent49[merged_latent49.index < pd.Period((str(1995)+"-1"),freq='M')]
df_val = merged_latent49[(merged_latent49.index >= pd.Period((str(1995)+"-1"),freq='M')) & (merged_latent49.index < pd.Period((str(2000)+"-1"),freq='M'))]
df_test = merged_latent49[merged_latent49.index >= pd.Period((str(2000)+"-1"),freq='M')]

merged_train_firm = df_train[firm_col]
merged_train_latent = df_train[latent_col]
merged_train_y = df_train['predicted_return']
print(merged_train_firm.shape, merged_train_latent.shape, merged_train_y.shape)

merged_val_firm = df_val[firm_col]
merged_val_latent = df_val[latent_col]
merged_val_y = df_val['predicted_return']
print(merged_val_firm.shape, merged_val_latent.shape, merged_val_y.shape)

merged_test_firm = df_test[firm_col]
merged_test_latent = df_test[latent_col]
merged_test_y = df_test['predicted_return']
print(merged_test_firm.shape, merged_test_latent.shape, merged_test_y.shape)

% -------------------------------------------------------------------------
def my_metric_fn(y_true, y_pred):
    num = tf.reduce_mean(tf.square(y_true - y_pred))
    den = tf.reduce_mean(tf.square(y_true))
    return 1 - num / den

def multi_input_model(num_input, dropout_, l1_reg_, l2_reg_):
    dropout = dropout_
    l1_reg = l1_reg_
    l2_reg =l2_reg_
    input1 = Input(shape=(num_input,))
    input2 = Input(shape=(num_input,))
    x1 = Dense(16, activation='relu')(input2)
    x1 = Dense(4, activation='relu')(x1)
    x1 = Dense(16, activation='relu')(x1)
    x1 = Dense(num_input, activation='relu')(x1)

    merged = Multiply()([input1, x1])
    x2 = Dense(32, activation='relu')(merged)
    x2 = Dense(16, activation='relu')(x2)
    #x2 = Dense(8, activation='relu')(x2)
    output = Dense(1, activation='linear')(x2)
    model = Model(inputs=[input1, input2], outputs=output)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, beta_1=0.92) 
    #model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=[my_metric_fn])
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[my_metric_fn])
    return model 

def Autoencoder1(num_input,lr=0.0001, acti='relu'):
    input_img = Input(shape=(num_input, ))
    encoded1 = Dense(16, activation=acti)(input_img)
    encoded2 = Dense(4, activation=acti)(encoded1)
    decoded2 = Dense(16, activation=acti)(encoded2)
    decoded1 = Dense(num_input, activation=acti)(decoded2)
    
    autoencoder = Model(inputs=input_img, outputs=decoded1)
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    autoencoder.compile(loss='mean_absolute_error',optimizer=opt, metrics=[my_metric_fn])
   #autoencoder.compile(loss='mean_squared_error', optimizer=opt, metrics=[my_metric_fn])
    return autoencoder

% -----------------------------------------------------------
## Conditional Autoencoder
import random 
seed_list = [458, 165, 530, 564, 590, 560, 829, 170, 376, 176]
yhat_df = pd.DataFrame()

for random_seed in seed_list:
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    autoencoder = Autoencoder1(num_input=merged_train_latent.shape[1])
    autoencoder.fit(merged_train_latent, merged_train_latent, 
                epochs = 20, batch_size = 216,
                validation_data = (merged_val_latent, merged_val_latent),
                 callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',mode='min', patience=1)])
 
    model = multi_input_model(num_input=merged_train_latent.shape[1], 
                              dropout_=0.2, 
                              l1_reg_=0.01, 
                              l2_reg_=0.01,)
    model.layers[0].set_weights(autoencoder.layers[0].get_weights())
    model.layers[2].set_weights(autoencoder.layers[2].get_weights())
    model.layers[0].trainable = False
    model.layers[2].trainable = False
    model.fit([merged_train_latent, merged_train_firm], merged_train_y,
                   epochs=50,
                   validation_data=([merged_val_latent, merged_val_firm], merged_val_y),
                   callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',mode='min', patience=1)])
    print(model.evaluate([merged_test_latent, merged_test_firm], merged_test_y))
    y_hat = model.predict([merged_test_latent, merged_test_firm]).reshape(-1)
    yhat_df[random_seed] = y_hat
    print()



class DataPrepare():
  def __init__(self):
    self.char_core = ['acc','agr','beta','bm','cash','cashpr','cfp','chatoia','chcsho','chfeps','chinv',
       'chmom','chpmia','chtx','currat','depr','dy','ear','ep','gma','grcapx','grltnoa',
       'ill','indmom','invest','lev','lgr','maxret','mom12m','mom1m','mom36m','mve','nincr',
       'orgcap','pchgm_pchsale','pchsale_pchinvt','pchsale_pchrect','pchsale_pchxsga',
       'retvol','roaq','roavol','roeq','salecash','saleinv','sgr','sp','std_dolvol','std_turn','turn']
    self.info_list = ['fyear','year','jyear','permno','ticker','comnam','exchcd','exchname','siccd',
       'indname','size_class','mve_m','rf','ret','ret_adj','ret_ex','ret_adj_ex',]
    self.macro_col = ['RPI', 'W875RX1', 'DPCERA3M086SBEA', 'CMRMTSPLx', 'RETAILx','INDPRO', 'IPFPNSS', 'IPFINAL', 'IPCONGD', 'IPDCONGD', 'IPNCONGD',
       'IPBUSEQ', 'IPMAT', 'IPDMAT', 'IPNMAT', 'IPMANSICS', 'IPB51222S','IPFUELS', 'CUMFNS', 'HWI', 'HWIURATIO', 'CLF16OV', 'CE16OV',
       'UNRATE', 'UEMPMEAN', 'UEMPLT5', 'UEMP5TO14', 'UEMP15OV','UEMP15T26', 'UEMP27OV', 'CLAIMSx', 'PAYEMS', 'USGOOD',
       'CES1021000001', 'USCONS', 'MANEMP', 'DMANEMP', 'NDMANEMP','SRVPRD', 'USTPU', 'USWTRADE', 'USTRADE', 'USFIRE', 'USGOVT',
       'CES0600000007', 'AWOTMAN', 'AWHMAN', 'HOUST', 'HOUSTNE','HOUSTMW', 'HOUSTS', 'HOUSTW', 'PERMIT', 'PERMITNE', 'PERMITMW',
       'PERMITS', 'PERMITW', 'AMDMNOx', 'ANDENOx', 'AMDMUOx', 'BUSINVx','ISRATIOx', 'M1SL', 'M2SL', 'M2REAL', 'BOGMBASE', 'TOTRESNS',
       'NONBORRES', 'BUSLOANS', 'REALLN', 'NONREVSL', 'CONSPI', 'S&P 500','S&P: indust', 'S&P div yield', 'S&P PE ratio', 'FEDFUNDS',
       'CP3Mx', 'TB3MS', 'TB6MS', 'GS1', 'GS5', 'GS10', 'AAA', 'BAA','COMPAPFFx', 'TB3SMFFM', 'TB6SMFFM', 'T1YFFM', 'T5YFFM', 'T10YFFM',
       'AAAFFM', 'BAAFFM', 'TWEXAFEGSMTHx', 'EXSZUSx', 'EXJPUSx','EXUSUKx', 'EXCAUSx', 'WPSFD49207', 'WPSFD49502', 'WPSID61',
       'WPSID62', 'OILPRICEx', 'PPICMM', 'CPIAUCSL', 'CPIAPPSL','CPITRNSL', 'CPIMEDSL', 'CUSR0000SAC', 'CUSR0000SAD',
       'CUSR0000SAS', 'CPIULFSL', 'CUSR0000SA0L2', 'CUSR0000SA0L5','PCEPI', 'DDURRG3M086SBEA', 'DNDGRG3M086SBEA', 'DSERRG3M086SBEA',
       'CES0600000008', 'CES2000000008', 'CES3000000008', 'UMCSENTx','DTCOLNVHFNM', 'DTCTHFNM', 'INVEST', 'VIXCLSx', 'dp', 'ep_macro', 'b/m',
       'ntis', 'tbl', 'tms', 'dfy', 'svar']

  def datasplit(self, df, dataset):
    df_train = df[df.index < pd.Period((str(1995)+"-1"),freq='M')]
    df_val = df[(df.index >= pd.Period((str(1995)+"-1"),freq='M')) & (df.index < pd.Period((str(2000)+"-1"),freq='M'))]
    df_test = df[df.index >= pd.Period((str(2000)+"-1"),freq='M')]
    if dataset == "firm":
      X_train = df_train[self.char_core]
      X_val = df_val[self.char_core]
      X_test = df_test[self.char_core]
    elif dataset == 'macro':
      X_train = df_train[self.macro_col]
      X_val = df_val[self.macro_col]
      X_test = df_test[self.macro_col]
    elif dataset == "firm_macro":
      X_train = df_train[self.char_core+self.macro_col]
      X_val = df_val[self.char_core+self.macro_col]
      X_test = df_test[self.char_core+self.macro_col]
    y_train = df_train['predicted_return']
    y_val = df_val['predicted_return']
    y_test = df_test['predicted_return']
    return (X_train, y_train, X_val, y_val, X_test, y_test, )


# HyperParameter Tuning
class MyHyperModel(HyperModel):
  def __init__(self, input_shape, l1_reg, l2_reg, layer, dropout, lr, batch_size, random_seed=None,):
    self.input_value = input_shape
    self.l1_reg = l1_reg
    self.l2_reg = l2_reg
    self.layer = layer
    self.dropout = dropout
    self.lr = lr
    self.batch_size = batch_size
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
    l1_reg = hp.Choice("l1_ratio", self.l1_reg)
    l2_reg = hp.Choice("l2_ratio", self.l2_reg)
    layer = hp.Choice("layer", self.layer)
    dropout = hp.Choice("dropout", self.dropout)
    lr = hp.Choice("learning_rate", self.lr)
    model = self.call_existing_code(
        l1_reg=l1_reg, l2_reg=l2_reg, L=layer, dropout=dropout, lr=lr
    )
    return model

  def fit(self, hp, model, *args, **kwargs):
    return model.fit(
    *args,
    batch_size=hp.Choice("batch_size", self.batch_size),
    epochs=self.epoch,
    **kwargs,
    )

# Import Dataset
path = '/content/drive/MyDrive/Colab Notebooks/Paper1/'
firm_df = pd.read_pickle(path+'firm_df49.pkl')
firm_df.index = pd.to_datetime(firm_df.index).to_period('M')

# macro-economic variables
macro_df = pd.read_pickle(path+'macro_df.pkl')
macro_df.index = pd.to_datetime(macro_df.index, format='%m/%d/%Y')
macro_df.rename(columns={"ep": "ep_macro"}, inplace=True)
macro_index = macro_df.index[1:]
macro_new = macro_df.iloc[:-1, :]
macro_new.index = macro_index
macro_new.index = macro_new.index.to_period('M')

# merged data 1
merged_df = pd.merge(firm_df, macro_new, left_on='jdate', right_on='sasdate', how='inner')
merged_df.index = firm_df.index

prepare = DataPrepare()
X_train, y_train, X_val, y_val, X_test, y_test = prepare.datasplit(firm_df, dataset = "firm")
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)




random_seed = 1120
tuner = kt.RandomSearch(
    MyHyperModel(input_shape=X_train.shape[1],
                 l1_reg = [0.,],
                 l2_reg = [0.1,],
                 layer = [4],
                 dropout = [0.,],
                 lr = [0.0001],
                 batch_size = [128,],
                 random_seed=random_seed,
                 ),
    objective="val_loss",

    max_trials=100,
    directory="my_dir",
    overwrite=True,
    project_name="tune_hypermodel",
)
tuner.search(X_train, y_train, validation_data=(X_val, y_val),
       callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)])

# Print best parameters
best_hps = tuner.get_best_hyperparameters(5)
hps1_dic = best_hps[0].values
print(hps1_dic)


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
    best_model.fit(x=x_all, y=y_all, epochs=20,
              callbacks=[keras.callbacks.EarlyStopping(monitor='loss',mode='min', patience=2)])
    print(best_model.evaluate(X_test,y_test))
    y_hat = best_model.predict(X_test).reshape(-1)
    yhat_df[random_seed] = y_hat
    print()



# Print out Predictive R^2
y_predict = yhat_df.mean(axis=1).values.reshape(-1)
y_real = y_test
a = np.mean(np.square(y_predict -  y_real))
b = np.mean(np.square(y_real))
print(1-a/b) ### 0.007397410372446012

