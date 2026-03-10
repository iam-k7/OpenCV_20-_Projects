# Train the model using the created dataset

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()  # Initializing the CNN

model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)


training_set = train_datagen.flow_from_directory(
    "dataset/train", target_size=(64, 64), batch_size=32, class_mode="binary"
)
val_set = val_datagen.flow_from_directory(
    "dataset/val", target_size=(64, 64), batch_size=32, class_mode="binary"
)

model.fit(
    training_set,
    steps_per_epoch=100,
    epochs=10,
    validation_data=val_set,
    validation_steps=2,
)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Model trained and saved to disk")
