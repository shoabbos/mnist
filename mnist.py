import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import keras
import numpy as np
from PIL import Image

img_rows, img_cols = 28, 28
#загрузка данных от дата сет мнист
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


image_index = 12 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
plt.imshow(x_train[image_index], cmap='Greys')

x_train.shape

# Преобразование массива в 4-dims, чтобы он мог работать с API Keras
x_train = x_train.reshape (x_train.shape [ 0 ], img_rows, img_cols , 1 )
x_test = x_test.reshape (x_test.shape [ 0 ], img_rows, img_cols , 1 )
input_shape = ( 28 , 28 , 1 )

x_test = np.concatenate((x_test, x_train[48000:60000, :, :, :]), axis=0)
x_train = x_train[0:60000, :, :, :]

y_test = np.concatenate((y_test, y_train[48000:60000]), axis=0)
y_train = y_train[0:60000]
# Убедиться, что значения являются плавающими, чтобы мы могли получить десятичные точки после деления
x_train = x_train.astype ( 'float32' )
x_test = x_test.astype ( 'float32' )
# Нормализация кодов RGB путем деления его на максимальное значение RGB.
x_train /=  255
x_test /=  255
print ( ' x_train shape: ' , x_train.shape)
print ( ' Количество изображений в x_train ' , x_train.shape [ 0 ])
print ( ' Количество изображений в x_test ' , x_test.shape [ 0 ])


# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=20)

model.evaluate (x_test, y_test)
model_json = model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model/model.h5")
print("Модель сохранена")

image_index = 9000
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
print(pred.argmax())



