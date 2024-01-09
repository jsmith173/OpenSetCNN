#
tf.random.set_seed(0)
deterministic_model = nn.get_deterministic_model(
	input_shape=input_shape,
	units=num_of_classes
	loss="categorical_crossentropy",
	optimizer="adam",
	metrics=["accuracy"]
)
deterministic_model.summary()
deterministic_model.fit(x_train_subset, y_train_subset, batch_size=128, epochs=5, validation_split=0.1)
deterministic_model.save('deterministic_mnist.keras')  

print('Accuracy on MNIST test set: ', str(deterministic_model.evaluate(x_test, y_test, verbose=False)[1]))



 value = i
 x_train_subset = nn.get_subset(x_train, y_train, value)
 len_subset = len(x_train_subset)
 len_train = int(0.8*len_subset)
 x_split = np.split(x_train_subset, [len_train, len_subset])
 x_train = x_split[0]; x_test = x_split[1]; l1 = len(x_train); l2 = len(x_test)

 y_train_subset = np.zeros((len(x_train_subset),), dtype=int)
 y_train_subset.fill(value)

 y_train = keras.utils.to_categorical(y_train, num_classes)
 y_test = keras.utils.to_categorical(y_test, num_classes)
 print(x_train.shape); print(y_train.shape); print(x_test.shape); print(y_test.shape);

