print("dev was here")
#Neural_Network_Num_Recognizer
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#print(tf.__version__)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
	
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
# used to set values from 0 to 1

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 3)

val_loss = model.evaluate(x_test, y_test)
val_accuracy = model.evaluate(x_test, y_test)
#print(val_loss)
#print(val_accuracy)

#plt.imshow(x_train[1], cmap = plt.cm.binary)
#plt.show()
#print(x_train[1])

model.save('my_first_NN_reader_pogg.model')
testModel = tf.keras.models.load_model('my_first_NN_reader_pogg.model')
predict = testModel.predict([x_test])
print(predict)


print(np.argmax(predict[12]))

plt.imshow(x_test[12], cmap = plt.cm.binary)
plt.show()