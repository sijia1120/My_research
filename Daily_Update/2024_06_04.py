

# ---- Import Dataset ---------
#merged_latent.to_pickle('merged_latent49.pkl')
import pandas as pf
merged_latent = pd.read_pickle("/content/drive/MyDrive/PhD_博士主业/Paper_III_Nuclear_Norm/Code/Dataset/merged_latent49.pkl")

# merged data 1
firm_col = merged_latent.columns.tolist()[18:18+49]
latent_col = merged_latent.columns.tolist()[18+49+1:]
print(len(firm_col), len(latent_col))

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



# ---------------------- Define the Model --------------------------------
class ResidualBlock(tf.keras.layers.Layer):
  def __init__(self, n_layers, n_neurons, **kwargs):
    super().__init__(**kwargs)
    self.hidden = [tf.keras.layers.Dense(n_neurons, activation = "relu",
                                         kernel_initializer = "he_normal")
                   for _ in range(n_layers)]

  def call(self, inputs):
    Z = inputs
    for layer in self.hidden:
      Z = layer(Z)
    return inputs + Z



class ReconstructingRegressor(keras.models.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = tf.keras.layers.Dense(49,activation="selu",kernel_initializer="lecun_normal")
        self.block1 = ResidualBlock(n_layers=2, n_neurons = 49)
        self.block2 = ResidualBlock(n_layers=2, n_neurons = 49)
        self.out = keras.layers.Dense(output_dim, name="output")

    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = keras.layers.Dense(n_inputs, name="reconstruct")
        super().build(batch_input_shape)

    @staticmethod
    def reconstruction_loss(reconstruction, inputs, rate=0.05):
        return tf.reduce_mean(tf.square(reconstruction - inputs)) * rate

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred, recon = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
            loss += self.reconstruction_loss(recon, x)
        gradients = tape.gradient(loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=None):
        Z = inputs
        for _ in range(1+2):
          Z = self.block1(Z)
        Z= self.block2(Z)

        if training:
            return self.out(Z), self.reconstruct(Z)



import random
import keras

merged_train_latent = merged_train_latent.astype(np.float32)
merged_train_y = merged_train_y.astype(np.float32)

merged_val_latent = merged_val_latent.astype(np.float32)
merged_val_y = merged_val_y.astype(np.float32)

merged_test_latent = merged_test_latent.astype(np.float32)
merged_test_y = merged_test_y.astype(np.float32)

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

    model = ReconstructingRegressor(output_dim = 1)
    model.compile(optimizer="nadam", loss="mse")
    history = model.fit(x=merged_train_latent,
                        y=merged_train_y,
                        validation_data=(merged_val_latent, merged_val_y),
                        epochs=30,
                        batch_size = 128,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',mode='min', patience=1)])
    print(model.evaluate(merged_test_latent, merged_test_y))
    y_hat = model.predict(merged_test_latent).reshape(-1)
    yhat_df[random_seed] = y_hat
    print()

# Print out Predictive R^2
y_predict = yhat_df.mean(axis=1).values.reshape(-1)
y_real = merged_test_y.values

a = np.mean(np.square(y_predict -  y_real))
b = np.mean(np.square(y_real))
print(1-a/b) ----> 0.1692706926293679



# -----------------------------------------------------------



