## List of 88 Firm-level Charactersitics and 49 core variable ##

#char_core =['acc', 'agr', 'beta', 'bm', 'cash','cashpr', 'cfp','chatoia', 'chcsho','chfeps','chinv', 'chmom','chpmia', 'chtx',
#'mom36m','mve','nincr','orgcap','pchgm_pchsale','pchsale_pchinvt', 'pchsale_pchrect', 'pchsale_pchxsga','retvol', 'roaq', 
#'currat', 'depr','dy', 'ear', 'ep', 'gma','grcapx', 'grltnoa','ill', 'indmom', 'invest','lev', 'lgr', 'maxret', 'mom12m', 'mom1m',
#'roavol', 'roeq','salecash', 'saleinv','sgr','sp', 'std_dolvol', 'std_turn', 'turn',]
# some features are deleted, new char_core is shown as follow (in total 43, 6 less than original 49 core features)
char_core =['acc', 'agr', 'beta', 'bm', 'cash','cashpr', 'cfp','chatoia', 'chcsho','chinv', 'chmom','chpmia', 'chtx','currat', 'depr','dy', 'ear', 'ep', 'gma',
            'grcapx', 'grltnoa','indmom', 'invest','lev', 'lgr', 'mom12m', 'mom1m','mom36m','mve','nincr','orgcap','pchgm_pchsale','pchsale_pchinvt', 'pchsale_pchrect', 
            'pchsale_pchxsga', 'roaq', 'roavol', 'roeq','salecash', 'saleinv','sgr','sp', 'turn',]


char_all = ['absacc','acc','aeavol','age','agr','beta','betasq','bm','bm_ia','cash','cashdebt','cashpr','cfp','cfp_ia','chatoia','chcsho','chempia','chinv','chmom','chpmia',
 'chtx','cinvest','convind','currat','depr','divi','divo','dolvol','dy','ear','egr','ep','gma','grcapx','grltnoa','herf','hire','idiovol','indmom',
 'invest','ipo','lev','lgr','mom12m','mom1m','mom36m','mom6m','ms','mve','mve_ia','nincr','operprof','orgcap','pchcapx_ia','pchcurrat','pchdepr',
 'pchgm_pchsale','pchquick','pchsale_pchinvt','pchsale_pchrect','pchsale_pchxsga','pchsaleinv','pctacc','pricedelay','ps','quick','rd','rd_mve','rd_sale',
 'roaq','roavol','roeq','roic','rsup','salecash','saleinv','salerec','securedind','sgr','sgrvol','sin','sp','stdacc','stdcf','sue','tang',
 'tb','turn']
print(len(char_core), len(char_all))


#%% Import Libraries 
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


#%% Predefined Functions 
def r2_oos(y_pred, y_real):
    return 1 - np.sum((y_pred - y_real)**2) / np.sum((y_real)**2)


# %% Import Dataset 
def get_Dataset(df_file):
    firm_data = pd.read_pickle(df_file)
    firm_data_new = firm_data.copy()
    firm_data_new['date'] = firm_data_new.index
    firm_data_new['date'] = pd.to_datetime(firm_data_new['date'])
    firm_data_new.reset_index(inplace=True)
    features = firm_data_new.columns[16:-2].tolist()
    target = ['predicted_return']
    all_cols =['date','permno'] + features + target
    data_ml = firm_data_new[all_cols]
    return data_ml
data_ml = get_Dataset(df_file = '/kaggle/input/year2-phd-dataset/firm_df.pkl')


# %% Features and Target (for 88 variables) ----------------
target = ['predicted_return']
features = data_ml.columns[2:-1].tolist()
# for 43 core variable 
data_ml = data_ml[['date','permno','predicted_return'] + char_core]
features = char_core


# %% Split the dataset for linear regression ----------------
train_set_num = int(data_ml['date'].nunique()*0.7)
val_set_num = int(data_ml['date'].nunique()*0.1)

train_split_date = data_ml['date'].unique()[train_set_num]
vali_split_date = data_ml['date'].unique()[train_set_num + val_set_num]
print("Train Set: ",train_set_num,train_split_date)
print("Validation Set: ",vali_split_date)

separation_date = vali_split_date
idx_train = data_ml.index[data_ml['date'] < separation_date].tolist()
idx_test = data_ml.index[data_ml['date'] >= separation_date].tolist()
data_ml_train = data_ml.loc[idx_train]
data_ml_test = data_ml.loc[idx_test]

y_penalized_train = data_ml_train[target].values
X_penalized_train = data_ml_train[features].values
y_penalized_test = data_ml_test[target].values
X_penalized_test = data_ml_test[features].values



#%% Fit the Lasso model and print R2 ---------------
def fit_model(train_set, test_set, Model, alpha):
    x_penalized_train = train_set[features].values 
    y_penalized_train = train_set[target].values 
    x_penalized_test = test_set[features].values
    model = Model(alpha=alpha)
    model.fit(x_penalized_train, y_penalized_train)
    model_coef = model.coef_
    y_pred = model.predict(x_penalized_test).reshape(-1)
    y_real = test_set[target].values.reshape(-1)
    mse = mean_squared_error(y_pred=y_pred, y_true=y_real)
    r2 = r2_oos(y_pred, y_real)
    return model_coef, mse,r2

#%% Lasso Model Find Best Alpha 
Model = Lasso
alphas = np.arange(1e-5, 1.0e-3, 1e-5)
results = {}
for alpha in alphas:
    model_coef, mse, r2 = fit_model(data_ml_train, data_ml_test, Lasso, alpha)
    print(f"alpha: {alpha}, mse: {mse}, r2: {r2}")
    results[alpha] = {"coefficent":model_coef,"mse": mse, "r2": r2}
    
#%% Plot the R2 and MSE of Lasso Model 
results_df = pd.DataFrame(results).T
results_df.to_csv("results_df_Lasso.csv")
results_df = pd.read_csv("/kaggle/input/aanalysis-result/results_df_Lasso.csv", index_col="Unnamed: 0")
plt.figure(figsize=(13, 8))
plt.plot(results_df.index, results_df['r2']*100)
plt.xlabel('Lambda')
plt.ylabel('Lasso $R^2$ (%)')
plt.show()

#%% Find the index (Lambda) with the highest 'r2'
max_r2_lambda = results_df['r2'].idxmax()
max_r2_value = results_df['r2'].max()
print("Highest r2 value:", max_r2_value)
print("Corresponding Lambda:", max_r2_lambda)
# ----- Highest r2 value: 0.0010169687171329
# ----- Corresponding Lambda: 0.00037



#%% Plot the Variable Importance Beta 
Lasso_coefficient_list = []
for i in range(len(results_df)):
    str_list = results_df.iloc[i][0]
    str_list= str_list.strip('[]') 
    coefficients_list = [float(val) for val in str_list.split()]
    Lasso_coefficient_list.append(coefficients_list)

df_lasso_res = pd.DataFrame(Lasso_coefficient_list,columns = char_all)
df_lasso_res.head()
predictors =(df_lasso_res.abs().sum() > 0.08)
# selecting the most relevant 
df_lasso_res.loc[:,predictors].plot( xlabel='Lambda',ylabel='Beta',figsize=(12,8)); # Plot!
