from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np

# ------ Extract Dataset -------------------------
def preprocess_df(X_df, y_df):
  y_df['date'] = X_df['date']
  print(X_df.shape, y_df.shape)

  X_df["date"] = pd.to_datetime(X_df["date"])
  X_df['year'] = X_df['date'].dt.year
  y_df['year'] = X_df["year"]
  print(X_df['year'].nunique(), X_df['date'].nunique())
  return X_df, y_df 

file_path = "/content/drive/MyDrive/PhD_博士主业/Paper_II/Coding/Dixon_NN/"
X_df = pd.read_csv(file_path + "X.csv")
y_df = pd.read_csv(file_path + "Y.csv")
X_df, y_df = preprocess_df(X_df, y_df)

training_periods = 330 
n = training_periods = 330 # number of model training and evaluation periods
L=2 # Number of hidden layers
n = y_df['date'].nunique()
training_dates = y_df['date'].unique()[(n-training_periods)-1:n-1] # leave the last period for test set


# ------------ Lasso "alpha" parameter tuning ---------------
train_idx ='2015-07-01'
test_idx = '2015-08-01'
x_train_tune = X_df[X_df['date']== train_idx].drop(columns = ["date","year"]).values 
y_train_tune = y_df[y_df['date']== train_idx].drop(columns = ["date","year"]).values
x_test_tune = X_df[X_df['date']== test_idx].drop(columns = ["date","year"]).values
y_test_tune = y_df[y_df['date']== test_idx].drop(columns = ["date","year"]).values

#%% Predefined Functions 
def r2_oos(y_pred, y_real):
    return 1 - np.sum((y_pred - y_real)**2) / np.sum((y_real)**2)

def fit_model(x_train_tune, y_train_tune, x_test_tune,y_test_tune, Model, alpha):
    model = Model(alpha=alpha)
    model.fit(x_train_tune, y_train_tune)
    model_coef = model.coef_
    y_pred = model.predict(x_test_tune).reshape(-1)
    y_real = y_test_tune.reshape(-1)
    mse = mean_squared_error(y_pred=y_pred, y_true=y_real)
    r2 = r2_oos(y_pred, y_real)
    return model_coef, mse,r2

def Lasso_tune(x_train_tune, y_train_tune, x_test_tune, y_test_tune):
  Model = Lasso
  alphas = np.arange(1e-4, 1e-1, 1e-4)
  results = {}
  for alpha in alphas:
    model_coef, mse, r2 = fit_model(x_train_tune, y_train_tune, x_test_tune, y_test_tune, Lasso, alpha)
    #print(f"alpha: {alpha}, mse: {mse}, r2: {r2}")
    results[alpha] = {"coefficent":model_coef,"mse": mse, "r2": r2}
  min_mse_lambda = results_df['mse'].astype(float).idxmin()
  print("Corresponding Lambda:", min_mse_lambda)
  return min_mse_lambda

best_lasso = Lasso_tune(x_train_tune, y_train_tune, x_test_tune, y_test_tune)
print(best_lasso)

#%% ----- Plot the R2 and MSE of Lasso Model 
results_df = pd.DataFrame(results).T
#results_df.to_csv("results_df_Lasso.csv")
#results_df = pd.read_csv("/kaggle/input/aanalysis-result/results_df_Lasso.csv", index_col="Unnamed: 0")
plt.figure(figsize=(10, 6))
plt.plot(results_df.index, results_df['mse']*100)
plt.xlabel('Lambda')
plt.ylabel('Lasso $MSE$ (%)')
plt.show()

#%% Find the index (Lambda) with the highest 'r2'
min_mse_lambda = results_df['mse'].astype(float).idxmin()
min_mse_value = results_df['r2'].max()
print("Highest r2 value:", min_mse_value)
print("Corresponding Lambda:", min_mse_lambda)

max_r2_lambda = results_df['mse'].astype(float).idxmin()
max_r2_value = results_df['r2'].max()
print("Highest r2 value:", max_r2_value)
print("Corresponding Lambda:", max_r2_lambda)



# -------- Model Define and Training ---------------------------------------------
def LinearModel(X_train, y_train):
  linearmodel = LinearRegression()
  linearmodel.fit(X_train, y_train)
  return linearmodel

def LassoModel(X_train, y_train, alpha):
  lassomodel = Lasso(alpha=alpha)
  lassomodel.fit(X_train, y_train)
  return lassomodel

def RidgeModel(X_train, y_train):
  ridgemodel = Ridge(alpha=5000.0)
  ridgemodel.fit(X_train, y_train)
  return ridgemodel

def TrainTest_MonthIndex(date, X, Y, col =["date","year"]):
  train_index = train_index = Y[Y['date']==date].index

  if len(train_index)==0:
    return 
  date_next=pd.Timestamp(np.datetime64(date)).to_pydatetime() + relativedelta(months=+1)
  date_next = date_next.strftime('%Y-%m-%d')
  test_index  = Y[Y['date']==date_next].index
  if len(test_index)==0:
    return 
  x_train = X.loc[train_index]
  x_train=x_train.drop(columns = col, axis=1)
  y_train = Y.loc[train_index]
  y_train= y_train.drop(columns = col, axis=1)

  x_test  = X.loc[test_index]
  x_test =x_test.drop(columns = col, axis=1)
  y_test =  Y.loc[test_index]
  y_test=y_test.drop(columns = col, axis=1)
  #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
  return x_train.values, y_train.values, x_test.values, y_test.values

#%% Fit the Lasso model and print R2 ---------------
#%% Predefined Functions 
def r2_oos(y_pred, y_real):
    return 1 - np.sum((y_pred - y_real)**2) / np.sum((y_real)**2)

def fit_model(x_train_tune, y_train_tune, x_test_tune,y_test_tune, Model, alpha):
    model = Model(alpha=alpha)
    model.fit(x_train_tune, y_train_tune)
    model_coef = model.coef_
    y_pred = model.predict(x_test_tune).reshape(-1)
    y_real = y_test_tune.reshape(-1)
    mse = mean_squared_error(y_pred=y_pred, y_true=y_real)
    r2 = r2_oos(y_pred, y_real)
    return model_coef, mse,r2

def Lasso_tune(x_train_tune, y_train_tune, x_test_tune, y_test_tune):
  Model = Lasso
  alphas = np.arange(1e-3, 1.0, 1e-3)
  results = {}
  for alpha in alphas:
    model_coef, mse, r2 = fit_model(x_train_tune, y_train_tune, x_test_tune, y_test_tune, Lasso, alpha)
    #print(f"alpha: {alpha}, mse: {mse}, r2: {r2}")
    results[alpha] = {"coefficent":model_coef,"mse": mse, "r2": r2}
  min_mse_lambda = results_df['mse'].astype(float).idxmin()
  min_mse_value = results_df['mse'].astype(float).min()
  print("Corresponding Lambda:", min_mse_lambda,min_mse_value)
  return min_mse_lambda


# ---------- Model Training ----------------------------------------------
import timeit 
from dateutil.relativedelta import *
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error


i = 0
tune = False 
models = {}
models['Linear']=[]
models['Lasso']=[] 
models['Ridge'] = []
xs = {}
ys = {}
xs['train']=[]
xs['test']=[]
ys['train']=[]
ys['test']=[]


for date in training_dates:
  start_time = timeit.default_timer()
  print(i,date)
  x_train, y_train, x_test, y_test = TrainTest_MonthIndex(date, X= X_df, Y=y_df)

  n_inputs = x_train.shape[1]
  if n_inputs ==0:
    next

  if tune:
    print("cross-validation...")
    parameter_tuning(x_train, y_train, 3)
    tune=False
    
  linearmodel = LinearModel(X_train=x_train, y_train=y_train)
  #Best_alpha = Lasso_tune(x_train_tune=x_train, y_train_tune=y_train, x_test_tune=x_test, y_test_tune= y_test)
  lassomodel = LassoModel(X_train=x_train, y_train=y_train, alpha = 0.002)
  ridgemodel = RidgeModel(X_train=x_train, y_train=y_train)
  models["Linear"].append(linearmodel)
  models['Lasso'].append(lassomodel)
  models['Ridge'].append(ridgemodel)

  xs['train'].append(x_train)
  xs['test'].append(x_test)
  ys['train'].append(y_train)
  ys['test'].append(y_test)

  elapsed = timeit.default_timer() - start_time
  print("Elapsed time:" + str(elapsed) + " (s)")
  print()
  i+=1



# ----- Plot the error norms over time -------------
# ------- MSE Out-of-sample Linear -----------------
MSE_array_linear=np.array([0]*training_periods, dtype='float64')
MSE_array_lasso=np.array([0]*training_periods, dtype='float64')
MSE_array_ridge=np.array([0]*training_periods, dtype='float64')
y_hat_linear=[]
y_hat_lasso=[]
y_hat_ridge=[]
for i in range(training_periods):
    y_hat_linear.append(models['Linear'][i].predict(xs['test'][i]))
    y_hat_lasso.append(models['Lasso'][i].predict(xs['test'][i]))
    y_hat_ridge.append(models['Ridge'][i].predict(xs['test'][i]))
    MSE_test_linear= mean_squared_error(y_hat_linear[-1], ys['test'][i])
    MSE_test_lasso= mean_squared_error(y_hat_lasso[-1], ys['test'][i])
    MSE_test_ridge= mean_squared_error(y_hat_ridge[-1], ys['test'][i])
    #print(i,MSE_test_linear, MSE_test_lasso, MSE_test_ridge)
    MSE_array_linear[i]=MSE_test_linear
    MSE_array_lasso[i]=MSE_test_lasso
    MSE_array_ridge[i]=MSE_test_ridge

# ------- L Out-of-sample Linear -----------------
L_array_linear = np.zeros(training_periods, dtype='float64')
L_array_lasso = np.zeros(training_periods, dtype='float64')
L_array_ridge = np.zeros(training_periods, dtype='float64')

for i in range(training_periods):
    y_hat_linear.append(models['Linear'][i].predict(xs['test'][i]))
    y_hat_lasso.append(models['Lasso'][i].predict(xs['test'][i]))
    y_hat_ridge.append(models['Ridge'][i].predict(xs['test'][i]))
    L_inf_test_linear = np.max(np.abs(y_hat_linear[-1] - ys['test'][i]))
    L_inf_test_lasso = np.max(np.abs(y_hat_lasso[-1] - ys['test'][i]))
    L_inf_test_ridge = np.max(np.abs(y_hat_ridge[-1] - ys['test'][i]))

    L_array_linear[i] = L_inf_test_linear
    L_array_lasso[i] = L_inf_test_lasso
    L_array_ridge[i] = L_inf_test_ridge



import matplotlib.pyplot as plt
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
#NN_label='NN (' + str(round(MSE_NN,3)) +')'
#plt.plot(MSE_array_linear[:-1], color='red', label=NN_label)

OLS_label='OLS (' + str(round(np.mean(MSE_array_linear),3)) +')'
plt.plot(np.log(MSE_array_linear[:-1]), color='blue', label=OLS_label)

lasso_label='Lasso (' + str(round(np.mean(MSE_array_lasso),3)) +')'
plt.plot(np.log(MSE_array_lasso[:-1]), color='red', label=lasso_label)

ridge_label='Ridge (' + str(round(np.mean(MSE_array_ridge),3)) +')'
plt.plot(np.log(MSE_array_ridge[:-1]), color='green', label=ridge_label)

plt.ylabel('MSE (out-of-sample)')
plt.xlabel('dates')
plt.xticks(np.arange(0,330,12), training_dates[np.arange(0,330,12)], rotation=70)
plt.legend()
plt.savefig('MSE_error.eps', format='eps',dpi=1200,bbox_inches = "tight")


import matplotlib.pyplot as plt
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
#NN_label='NN (' + str(round(MSE_NN,3)) +')'
#plt.plot(MSE_array_linear[:-1], color='red', label=NN_label)

OLS_label='OLS (' + str(round(np.mean(L_array_linear),3)) +')'
plt.plot(np.log(L_array_linear[:-1]), color='blue', label=OLS_label)

lasso_label='Lasso (' + str(round(np.mean(L_array_lasso),3)) +')'
plt.plot(np.log(L_array_lasso[:-1]), color='red', label=lasso_label)

ridge_label='Ridge (' + str(round(np.mean(L_array_ridge),3)) +')'
plt.plot(np.log(L_array_ridge[:-1]), color='green', label=ridge_label)

plt.ylabel('$L_{\infty}$ Error (out-of-sample)')
plt.xlabel('dates')
plt.xticks(np.arange(0,330,12), training_dates[np.arange(0,330,12)], rotation=70)
plt.legend()
plt.savefig('MSE_error.eps', format='eps',dpi=1200,bbox_inches = "tight")



# -------- # Calculate information ratios ------------------------------
"""
 Information Ratio Calculation over most recent 10 year period
"""    
import random
testing_periods=training_periods

info_ratio = []
info_ratio_linear = []
info_ratio_lasso = []
info_ratio_ridge = []
info_ratio_wn = []

m_range = [10,15,20,25,30,35,40,45,50]
for m in m_range:
    excess_returns_linear = []
    excess_returns_lasso = []
    excess_returns_ridge = []
    excess_returns_wn = []
    excess_returns_std_linear = []
    excess_returns_std_lasso = []
    excess_returns_std_ridge = []
    excess_returns_std_wn = []
    idx_linear = []
    idx_lasso = []
    idx_ridge = []
    for i in range((testing_periods-120),testing_periods):
        idx_linear.append(np.argsort(-y_hat_linear[i].flatten())[:m])
        excess_returns_linear.append(np.mean(np.array(ys['test'][i])[idx_linear[-1]] ))
        
        idx_lasso.append(np.argsort(-y_hat_lasso[i].flatten())[:m])
        excess_returns_lasso.append(np.mean(np.array(ys['test'][i])[idx_lasso[-1]] ))

        idx_ridge.append(np.argsort(-y_hat_ridge[i].flatten())[:m])
        excess_returns_ridge.append(np.mean(np.array(ys['test'][i])[idx_ridge[-1]] ))
          
        #White Noise IR
        wn = np.asarray(random.sample(range(1, len(ys['test'][i])),m))
        excess_returns_wn.append(np.mean(np.array(ys['test'][i])[wn]))
        
    excess_returns_std_linear.append(np.std(excess_returns_linear))
    info_ratio_linear.append(np.mean(excess_returns_linear)/excess_returns_std_linear)
    excess_returns_std_lasso.append(np.std(excess_returns_lasso))
    info_ratio_lasso.append(np.mean(excess_returns_lasso)/excess_returns_std_lasso)
    excess_returns_std_ridge.append(np.std(excess_returns_ridge))
    info_ratio_ridge.append(np.mean(excess_returns_ridge)/excess_returns_std_ridge)
    excess_returns_std_wn.append(np.std(excess_returns_wn))
    info_ratio_wn.append(np.mean(excess_returns_wn)/excess_returns_std_wn)



wn_label='Random (' + str(round(np.mean(info_ratio_wn),3)) + ')'
plt.plot(m_range, info_ratio_wn, label=wn_label, color='black')
OLS_label='OLS (' + str(round(np.mean(info_ratio_linear),3)) + ')'
plt.plot(m_range, info_ratio_linear, label=OLS_label, color='blue')
lasso_label='Lasso (' + str(round(np.mean(info_ratio_lasso),3)) + ')'
plt.plot(m_range, info_ratio_lasso, label=lasso_label, color='red')
ridge_label='Ridge (' + str(round(np.mean(info_ratio_ridge),3)) + ')'
plt.plot(m_range, info_ratio_ridge, label=ridge_label, color='green')

#plt.plot(m_range, info_ratio_NN_relu, label='NN (ReLU 10)', color='red', linestyle='--')
plt.xlabel('Number of Stocks')
plt.ylabel('Information Ratio')
plt.legend()
plt.savefig('IR.eps', format='eps',dpi=1200,bbox_inches = "tight")


