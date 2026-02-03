from numpy import loadtxt
from keras.models import model_from_json


dataset = loadtxt('D:\OpenCV_20+_Projects\Day-06\pima-indians-diabetes.csv', delimiter=',')
x = dataset[:,0:8]    #input features
y = dataset[:,8]      #output label

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()       
model = model_from_json(loaded_model_json)
model.load_weights("diabetes_model.weights.h5")
print("Loaded model from disk")

predictions = model.predict(x)

for i in range(20, 25):
    print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))