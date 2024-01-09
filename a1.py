#
x_train = 1.0-x_train.astype("float32") / 255.0
x_test = 1.0-x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

tf.random.set_seed(0)
deterministic_model = nn.get_deterministic_model(
	input_shape=(28, 28, 1),
	loss="categorical_crossentropy",
	optimizer="adam",
	metrics=["accuracy"]
)
deterministic_model.summary()
deterministic_model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)
deterministic_model.save('deterministic_mnist.keras')  

print('Accuracy on MNIST test set: ', str(deterministic_model.evaluate(x_test, y_test, verbose=False)[1]))
