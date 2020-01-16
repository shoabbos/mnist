import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import keras
from keras.preprocessing import image
import numpy as np
from PIL import Image
from keras.models import model_from_json



Images = ["img/test0.png", "img/test1.png", "img/test2.png", "img/test3.png", "img/test4.png", "img/test5.png", "img/test6.png", "img/test7.png", "img/test8.png", "img/test9.png",]
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
inverse = True

model = Sequential()
model.add(Conv2D(28, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

model.load_weights("model//model.h5")
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
for number in range(10):
	print('Изображение ', number)
	img_path = Images[number]
	img = image.load_img(img_path, target_size=(28, 28), color_mode = "grayscale")
	plt.imshow(img.convert('RGBA'))
	data = image.img_to_array(img)
	data = data.reshape(1, 28, 28, 1)
	data /= 255
	if(inverse):
		data -= 1
		data *= -1

	prediction = model.predict(data)
	print('Вероятно', np.argmax(prediction))

