from numpy import loadtxt
from keras.models import Sequential, model_from_json
from keras.layers import Dense

dataset = loadtxt('D:\OpenCV_20+_Projects\Day-06\pima-indians-diabetes.csv', delimiter=',')
print(dataset)

x = dataset[:,0:8]    #input features
y = dataset[:,8]      #output label

print("Input", x)
print("Output", y)

model = Sequential()

model.add(Dense(12, activation='relu', input_shape=(8,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))        #sigmoid for binary classification(0 or 1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model Training
model.fit(x, y, epochs=12, batch_size=10)

# Evaluate the model
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))

# Save the model to JSON

# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# model.save_weights("diabetes_model.weights.h5")
# print("saved model to disk")