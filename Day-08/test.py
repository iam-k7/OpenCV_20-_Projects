from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("Loaded model from disk")

def classify(img_path):
    img_name = img_path
    test_time = image.load_img(img_name, target_size=(64, 64))

    test_image = image.img_to_array(test_time)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    if result[0][0] > 0.5:
        prediction = 'IronMan'
    else:
        prediction = 'SpiderMan'
    print(prediction, img_name)


import os 
path = r"D:\OpenCV_20+_Projects\Day-08\dataset\test"
files = []
# r = root, d = directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))

for img_file in files:
    classify(img_file)
