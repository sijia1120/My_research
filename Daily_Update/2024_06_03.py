# --- Residual Block with Reconstruction Loss ---------

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




num_training = 100
num_dim = 49

X = np.random.random((100, 49)).astype(np.float32)
y = np.random.random((100,)).astype(np.float32)


class ReconstructingRegressor(keras.models.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [
            keras.layers.Dense(
                30,
                activation="selu",
                kernel_initializer="lecun_normal",
                name=f"hidden_{idx}",
            )
            for idx in range(5)
        ]
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
        for layer in self.hidden:
            Z = layer(Z)
        if training:
            return self.out(Z), self.reconstruct(Z)

        return self.out(Z)


model = ReconstructingRegressor(output_dim = 1)

model.compile(optimizer="nadam", loss="mse")
history = model.fit(X, y, epochs=10)
history = model.evaluate(X, y)
