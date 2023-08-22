model = keras.models.Sequential()
model.add(TridiagonalSparseLayer(100, activation='relu', input_shape=(100,)))
model.add(TridiagonalSparseLayer(100, activation='relu'))
model.add(TridiagonalSparseLayer(100, activation='relu'))
model.add(TridiagonalSparseLayer(100, activation='relu'))
model.add(TridiagonalSparseLayer(100, activation='relu'))
model.add(TridiagonalSparseLayer(100, activation='relu'))
model.add(TridiagonalSparseLayer(100, activation='relu'))
model.add(TridiagonalSparseLayer(100, activation='relu'))
model.add(TridiagonalSparseLayer(100, activation='relu'))
model.add(TridiagonalSparseLayer(100, activation='relu'))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(inputs, outputs, epochs=3, batch_size=20)
model.summary()

# Test the model on some new inputs
test_inputs = np.random.rand(7000, 100)
# Predict outputs for test data
predicted_outputs = model.predict(test_inputs)
