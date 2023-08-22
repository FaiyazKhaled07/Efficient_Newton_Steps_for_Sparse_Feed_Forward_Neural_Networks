class TridiagonalInitializer(tf.keras.initializers.Initializer):
    def __init__(self, low=-1, high=1, seed=None):
        self.low = low
        self.high = high
        self.seed = seed

    def __call__(self, shape, dtype=None):
        size = shape[0]
        dtype = np.dtype(dtype.as_numpy_dtype)
        tridiag = np.zeros((size, size), dtype=dtype)
        np.fill_diagonal(tridiag, np.random.uniform(self.low, self.high, size).astype(dtype))
        np.fill_diagonal(tridiag[1:], np.random.uniform(self.low, self.high, size - 1).astype(dtype))
        np.fill_diagonal(tridiag[:, 1:], np.random.uniform(self.low, self.high, size - 1).astype(dtype))

        return tridiag


class TridiagonalSparseLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        super(TridiagonalSparseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mat = self.add_weight(name='mat',
                                   shape=(self.units, self.units),
                                   initializer=TridiagonalInitializer(),
                                   trainable=True)
        self.b = self.add_weight(name='b',
                                 shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
        super(TridiagonalSparseLayer, self).build(input_shape)

    def call(self, x):
      w = tf.where(tf.abs(self.mat) > 1e-5, self.mat, tf.zeros_like(self.mat))
      output = tf.matmul(x, w) + self.b
      if self.activation is not None:
        output = self.activation(output)
      return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
