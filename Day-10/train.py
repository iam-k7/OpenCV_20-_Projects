# leaf disease classification using CNN

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

#basic CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=None, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(r'D:\OpenCV_20+_Projects\Day-10\datasets\train',
                                                target_size=(128, 128),
                                                batch_size=32,
                                                class_mode='categorical')

labels = (training_set.class_indices)
print(labels)

test_set = test_datagen.flow_from_directory(r'D:\OpenCV_20+_Projects\Day-10\datasets\test',
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='categorical')

labels = (test_set.class_indices)
print(labels)

model.fit(training_set,
                    steps_per_epoch=375,
                    epochs=10,
                    validation_data=test_set,
                    validation_steps=125)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json) 
    model.save_weights("model.h5")
print("Saved model to disk")