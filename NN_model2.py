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
