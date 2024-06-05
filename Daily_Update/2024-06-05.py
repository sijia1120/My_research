% ---- Cutomized KL Divergence Penalty Term ------

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K
import keras

def kl_divergence_regularizer(inputs):
    # Ensure inputs has a single dimension
    if K.ndim(inputs) == 0:
        inputs = K.reshape(inputs, (1,))

    means = K.mean(inputs, axis=-1)  # Specify the reduction dimension (axis=-1 means reduce along the last axis)
    rho = 0.005
    down = 0.005 * K.ones_like(means)
    up = (1 - 0.005) * K.ones_like(means)
    return 0.5 * (0.01 * (keras.losses.kullback_leibler_divergence(down, means) + keras.losses.kullback_leibler_divergence(up, 1 - means)))


# Create the model
dnn = Sequential()
dnn.add(Dense(128, activation='relu', kernel_regularizer=kl_divergence_regularizer))
dnn.add(Dense(64, activation='relu', kernel_regularizer=kl_divergence_regularizer))
dnn.add(Dense(32, activation='relu', kernel_regularizer=kl_divergence_regularizer))
dnn.add(Dense(16, activation='relu', kernel_regularizer=kl_divergence_regularizer))
dnn.add(Dense(1))

# Compile the model
dnn.compile(optimizer='Adam', loss='mse')

# Define EarlyStopping callback
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)

# Train the model
dnn.fit(x=merged_train_latent,
        y=merged_train_y,
        validation_data=(merged_val_latent, merged_val_y),
        epochs=30,
        batch_size=128,
        callbacks=[callback])

# Display model summary
dnn.summary()

# Evaluate the model
y_pred = dnn.predict(merged_test_latent).reshape(-1)
y_real = merged_test_y.values
mse = np.mean(np.square(y_pred - y_real))
r_squared = 1 - mse / np.mean(np.square(y_real))
print("R^2 Score:", r_squared)
