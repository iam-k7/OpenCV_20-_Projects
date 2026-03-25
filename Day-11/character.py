# Character Recognition using KNN

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from pathlib import Path
from keras.preprocessing import image
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"
MODEL_JSON_PATH = BASE_DIR / "model.json"
MODEL_WEIGHTS_PATH = BASE_DIR / "model.h5"


def number_to_words(value):
    ones = [
        "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
        "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
        "Seventeen", "Eighteen", "Nineteen",
    ]
    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]

    if value < 20:
        return ones[value]

    tens_word = tens[value // 10]
    ones_value = value % 10
    if ones_value == 0:
        return tens_word
    return f"{tens_word} {ones[ones_value]}"


def class_name_to_english(class_name):
    if class_name.startswith("box_"):
        suffix = class_name.split("_", 1)[1]
        if suffix.isdigit():
            return f"Box {number_to_words(int(suffix))}"
    return class_name.replace("_", " ").title()


def sorted_class_names(directory):
    return sorted(
        [entry.name for entry in directory.iterdir() if entry.is_dir()],
        key=lambda name: int(name.split("_", 1)[1]) if name.startswith("box_") and name.split("_", 1)[1].isdigit() else name,
    )


def get_dataset_labels():
    train_classes = sorted_class_names(TRAIN_DIR)
    test_classes = sorted_class_names(TEST_DIR)
    if train_classes != test_classes:
        raise ValueError("Train and test class folders do not match in Day-11/dataset.")
    return [class_name_to_english(class_name) for class_name in train_classes]


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1150, 992)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(640, 640, 281, 71))
        self.textBrowser.setObjectName("textBrowser")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(260, 80, 641, 51))
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(360, 180, 451, 381))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(270, 640, 161, 71))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(270, 770, 161, 71))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(650, 770, 161, 71))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1150, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.BrowseImage = self.pushButton
        self.Classify = self.pushButton_2
        self.Training = self.pushButton_3
        self.imageLbl = self.label_2
        self.textEdit = self.textBrowser

        self.BrowseImage.clicked.connect(self.loadImage)
        self.Classify.clicked.connect(self.classifyFunction)
        self.Training.clicked.connect(self.trainingFunction)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "GUJARATI CHARACTER RECOGNITION USING CNN"))
        self.pushButton.setText(_translate("MainWindow", "Browse image"))
        self.pushButton_2.setText(_translate("MainWindow", "Classify"))
        self.pushButton_3.setText(_translate("MainWindow", "Training"))


    def loadImage(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if fileName:
            print(fileName)
            self.file=fileName
            pixmap = QtGui.QPixmap(fileName)
            pixmap = pixmap.scaled(self.imageLbl.width(), self.imageLbl.height(), QtCore.Qt.KeepAspectRatio)
            self.imageLbl.setPixmap(pixmap)
            self.imageLbl.setAlignment(QtCore.Qt.AlignCenter)

    def classifyFunction(self):
        if not hasattr(self, "file"):
            self.textBrowser.setText("Please browse and select an image first.")
            return

        if not MODEL_JSON_PATH.exists() or not MODEL_WEIGHTS_PATH.exists():
            self.textBrowser.setText("Model files not found in Day-11. Run Training first.")
            return

        label = get_dataset_labels()

        json_file = open(MODEL_JSON_PATH, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(str(MODEL_WEIGHTS_PATH))
        print("Loaded model from disk")
        path2 = self.file
        print(path2)
        test_image = image.load_img(path2, target_size=(128, 128), color_mode='categorical')
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = loaded_model.predict(test_image)

        fresult = np.max(result)
        label2 = label[result.argmax()]
        print(label2)
        self.textBrowser.setText(label2)


    def trainingFunction(self):
        self.textEdit.setText("Training started...")
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(len(get_dataset_labels()), activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        train_datagen = ImageDataGenerator(rescale=None,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)
        
        test_datagen = ImageDataGenerator(rescale=1./255)

        training_set = train_datagen.flow_from_directory(str(TRAIN_DIR),
                                                         target_size=(128, 128),
                                                         batch_size=8,
                                                         color_mode='grayscale',
                                                         class_mode='categorical')
        
        #print(test_datagen);
        labels = (training_set.class_indices)
        print({class_name_to_english(key): value for key, value in labels.items()})

        test_set = test_datagen.flow_from_directory(str(TEST_DIR),
                                                    target_size=(128, 128),
                                                    batch_size=8,
                                                    color_mode='grayscale',
                                                    class_mode='categorical')
        
        labels2 = (test_set.class_indices)
        print({class_name_to_english(key): value for key, value in labels2.items()})

        model.fit_generator(training_set,
                            steps_per_epoch = 100,
                            epochs = 10,
                            validation_data = test_set,
                            validation_steps = 125)
        
        # making new predictions

        # model_json = model.to_json()
        # with open(MODEL_JSON_PATH, "w") as json_file:
        #     json_file.write(model_json)
        # model.save_weights(str(MODEL_WEIGHTS_PATH))
        # print("Saved model to disk")
        # self.textEdit.setText("Training completed and model saved to disk.")



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())



