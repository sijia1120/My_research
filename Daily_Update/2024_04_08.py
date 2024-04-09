#%% Import dataset 
firm = pd.read_pickle("7_chars_data.pkl")
firm_collect = FirmCollect()
firm_data = firm.copy()
firm_data = firm_collect.replace_missing_median(firm_data)
firm_data_new = firm_collect.cs_Rank(firm_data)
firm_data_new = firm_collect.get_label(firm_data_new)
firm_data_new.to_pickle('firm_df.pkl')

# %% Features of 
features = firm_data_new.columns[17:]
target = ['predicted_return']


# %% There is no missing value in the dataset 
for i in features:
    if firm_data_new[i].isnull().sum() != 0:
        print(i, firm_data_new[i].isnull().sum())


#%% Shape of the Dataset 
firm_data_new.shape # (2007064, 121)
