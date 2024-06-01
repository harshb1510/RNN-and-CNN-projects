import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets,layers,models

(training_images,training_labels),(testing_images,testing_labels) = datasets.cifar10.load_data()
training_images,testing_images = training_images/255.0,testing_images/255.0

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])
plt.show()

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(training_images,training_labels,epochs=10,validation_data=(testing_images,testing_labels))
plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.legend(loc='lower right')
plt.show()
test_loss,test_acc = model.evaluate(testing_images,testing_labels,verbose=2)
print(test_acc)
model.save('image_classification.h5')
model = models.load_model('image_classification.h5')
predictions = model.predict(testing_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(testing_labels[0])
plt.imshow(testing_images[0],cmap=plt.cm.binary)
plt.show()

