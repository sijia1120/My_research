# Import Dataset
path = "dataset/"
firm_df = pd.read_pickle(path+'firm_df49.pkl')
firm_df.index = pd.to_datetime(firm_df.index).to_period('M')

# Construct Portfolio Factors 
q = 0.2
chars_data = firm_df.copy()
factors_df = chars_data.groupby('jdate').apply(
        lambda x: x[char_core].apply(lambda z: x[z >= z.quantile(1 - q)]['predicted_return'].mean() -
                    x[z <= z.quantile(q)]['predicted_return'].mean()))

# Scale macro-features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
factors_scaled = scaler.fit_transform(factors_df)
factors_scaled_df = pd.DataFrame(factors_scaled, columns=factors_df.columns)
factors_scaled_df.index = factors_df.index

# merged data 1
merged_latent = pd.merge(firm_df, factors_scaled_df, left_on='jdate', right_on=factors_scaled_df.index, how='inner')
merged_latent.index = firm_df.index

# merged data 1
firm_col = merged_latent.columns.tolist()[18:18+49]
latent_col = merged_latent.columns.tolist()[18+49+1:]
print(len(firm_col), len(latent_col))

# --------------------------------------
df_train = merged_latent[merged_latent.index < pd.Period((str(1995)+"-1"),freq='M')]
df_val = merged_latent[(merged_latent.index >= pd.Period((str(1995)+"-1"),freq='M')) & (merged_latent.index < pd.Period((str(2000)+"-1"),freq='M'))]
df_test = merged_latent[merged_latent.index >= pd.Period((str(2000)+"-1"),freq='M')]

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


# -----------------------------------------
def my_metric_fn(y_true, y_pred):
  num = tf.reduce_mean(tf.square(y_true - y_pred))
  den = tf.reduce_mean(tf.square(y_true))
  return 1 - num / den

def multi_input_model():
  dropout =0.1
  l1_reg = 0.0
  l2_reg =0.0
  # Define the first input with 134 features
  # input 1 is the 49 observed
  # input 2 is the 49 latent variable
  input1 = Input(shape=(49,))

  # Input 2
  input2 = Input(shape=(49,))
  x1 = Dense(4, activation='relu')(input2)
  x1 = Dense(49, activation='relu')(x1)

  # Multiply the outputs of the first branch and the second input element-wise
  merged = Multiply()([input1, x1])
  # Define output layer
  # Create the second branch of the network
  x2 = Dense(32, activation='relu')(merged)
  x2 = Dense(16, activation='relu')(x2)
  output = Dense(1, activation='linear')(x2)


  # Create the model with two inputs and one output
  model = Model(inputs=[input1, input2], outputs=output)
  # optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)
  optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, beta_1=0.92) # best
  model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[my_metric_fn])
  return model

# -----------------------------
print(merged_train_firm.shape, merged_train_latent.shape, merged_train_y.shape)
# Create the model
random_seed = 1120
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
model = multi_input_model()

# Fit the model to your training data
model.fit([merged_train_firm, merged_train_latent], merged_train_y,
          validation_data=([merged_val_firm, merged_val_latent], merged_val_y),
          epochs=100,
          batch_size=128,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',patience=3)],
          )  # You can adjust the number
model.evaluate([merged_test_firm, merged_test_latent], merged_test_y)

# --------------------
seed_list = [458, 165, 530, 564, 590, 560, 829, 170, 376, 176]
yhat_df = pd.DataFrame()
# 10 Trials for Prediction
merged_train_latent_all = np.concatenate((merged_train_latent, merged_val_latent))
merged_train_firm_all = np.concatenate((merged_train_firm, merged_val_firm))
merged_y_all = np.concatenate((merged_train_y, merged_val_y))

for random_seed in seed_list:
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    model = multi_input_model()
    model.fit([merged_train_firm, merged_train_latent], merged_train_y,
              validation_data=([merged_val_firm, merged_val_latent], merged_val_y),
              epochs=30,
              batch_size = 128,
              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',mode='min', patience=1)]
              )
    print(model.evaluate([merged_test_firm, merged_test_latent], merged_test_y))
    y_hat = model.predict([merged_test_firm, merged_test_latent]).reshape(-1)
    yhat_df[random_seed] = y_hat
    print()

# Print out Predictive R^2
y_predict = yhat_df.mean(axis=1).values.reshape(-1)
y_real = merged_test_y

a = np.mean(np.square(y_predict -  y_real))
b = np.mean(np.square(y_real))
print(1-a/b)

