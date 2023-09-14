from google.colab import drive
drive.mount('/content/drive')

# Import Library 
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

!pip install keras_tuner
import keras_tuner
from keras_tuner import HyperModel
import keras_tuner as kt


char_core = ['acc','agr','beta','bm','cash','cashpr','cfp','chatoia','chcsho','chfeps','chinv',
       'chmom','chpmia','chtx','currat','depr','dy','ear','ep','gma','grcapx','grltnoa',
       'ill','indmom','invest','lev','lgr','maxret','mom12m','mom1m','mom36m','mve','nincr',
       'orgcap','pchgm_pchsale','pchsale_pchinvt','pchsale_pchrect','pchsale_pchxsga',
       'retvol','roaq','roavol','roeq','salecash','saleinv','sgr','sp','std_dolvol','std_turn','turn']
info_list = ['fyear','year','jyear','permno','ticker','comnam','exchcd','exchname','siccd',
       'indname','size_class','mve_m','rf','ret','ret_adj','ret_ex','ret_adj_ex',]
macro_col = ['RPI', 'W875RX1', 'DPCERA3M086SBEA', 'CMRMTSPLx', 'RETAILx','INDPRO', 'IPFPNSS', 'IPFINAL', 'IPCONGD', 'IPDCONGD', 'IPNCONGD',
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


# Import Dataset
path = '/content/drive/MyDrive/Colab Notebooks/Paper1/'
firm_df = pd.read_pickle(path+'firm_df49.pkl')
firm_df.index = pd.to_datetime(firm_df.index).to_period('M')
# macro-economic variables
# There is one features which has 37 missing values but before year 1980
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
    elif dataset == "firm_macro":
      X_train = df_train[self.char_core+self.macro_col]
      X_val = df_val[self.char_core+self.macro_col]
      X_test = df_test[self.char_core+self.macro_col]
    y_train = df_train['predicted_return']
    y_val = df_val['predicted_return']
    y_test = df_test['predicted_return']
    return (X_train, y_train, X_val, y_val, X_test, y_test, )


prepare = DataPrepare()
X_train, y_train, X_val, y_val, X_test, y_test = prepare.datasplit(merged_df, dataset = "firm_macro")
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
merged_train_firm = X_train[char_core]
merged_train_macro = X_train[macro_col]
merged_train_y = y_train
print(merged_train_firm.shape, merged_train_macro.shape, merged_train_y.shape)

merged_val_firm = X_val[char_core]
merged_val_macro = X_val[macro_col]
merged_val_y = y_val
print(merged_val_firm.shape, merged_val_macro.shape, merged_val_y.shape)

merged_test_firm = X_test[char_core]
merged_test_macro = X_test[macro_col]
merged_test_y = y_test
print(merged_test_firm.shape, merged_test_macro.shape, merged_test_y.shape)



def my_metric_fn(y_true, y_pred):
  num = tf.reduce_mean(tf.square(y_true - y_pred))
  den = tf.reduce_mean(tf.square(y_true - tf.zeros_like(y_true)))
  return 1 - num / den


def multi_input_model():
  # Define the first input with 134 features
  input1 = Input(shape=(134,))
  # Define the first branch of the network
  x1 = Dense(128, activation='relu')(input1)
  x1 = Dense(64, activation='relu')(x1)
  x1 = Dense(49, activation='relu')(x1)
  # Define the second input with 49 features
  input2 = Input(shape=(49,))
  # Multiply the outputs of the first branch and the second input element-wise
  merged = Multiply()([x1, input2])
  # Create the second branch of the network
  x2 = Dense(32, activation='relu')(merged)
  x2 = Dense(16, activation='relu')(x2)
  output = Dense(1, activation='linear')(x2)
  # Create the model with two inputs and one output
  model = Model(inputs=[input1, input2], outputs=output)
  # Compile the model (you can adjust the optimizer and loss function as needed)
  model.compile(optimizer='adam', loss='mean_squared_error', metrics=[my_metric_fn])
  return model 

model = multi_input_model()
model.summary()


print(merged_train_firm.shape, merged_train_macro.shape, merged_train_y.shape)
# Create the model
random_seed = 1120
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
model = multi_input_model()

# Fit the model to your training data
model.fit([merged_train_macro, merged_train_firm], merged_train_y, 
          validation_data=([merged_val_macro, merged_val_firm], merged_val_y),
          epochs=100, 
          batch_size=128,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',patience=3)],
          )  # You can adjust the number 


seed_list = [458, 165, 530, 564, 590, 560, 829, 170, 376, 176]
yhat_df = pd.DataFrame()
# 10 Trials for Prediction
merged_train_macro_all = np.concatenate((merged_train_macro, merged_val_macro))
merged_train_firm_all = np.concatenate((merged_train_firm, merged_val_firm))
merged_y_all = np.concatenate((merged_train_y, merged_val_y))

for random_seed in seed_list:
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    model.fit(x=[merged_train_macro_all, merged_train_firm_all], y=merged_y_all, epochs=20,
              callbacks=[keras.callbacks.EarlyStopping(monitor='loss',mode='min', patience=1)])
    print(model.evaluate([merged_test_macro, merged_test_firm], merged_test_y))
    y_hat = model.predict([merged_test_macro, merged_test_firm]).reshape(-1)
    yhat_df[random_seed] = y_hat
    print()


# Print out Predictive R^2
y_predict = yhat_df.mean(axis=1).values.reshape(-1)
y_real = y_test

a = np.mean(np.square(y_predict -  y_real))
b = np.mean(np.square(y_real))
print(1-a/b) ## 0.006065208697421998
