import cv2
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage
import tensorflow.keras
import numpy as np


np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')


def preprocess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img,(224,224), interpolation=cv2.INTER_NEAREST)
    
    normal_img = (img.astype(np.float32) / 127.0) - 1
    
    ready_img = normal_img.reshape(1,224,224,3)
    
    return ready_img

def get_prediction(img,model):
    preprocessed_img = preprocess(img)
    pred = model.predict(preprocessed_img)
    good_percent = pred[0][0]
    bad_percent = pred[0][1]
    if good_percent>bad_percent:
        return 'good : confidence '+str(good_percent)+" %"
    else:
        return 'bad : confidence '+str(bad_percent)+" %"

frame_rate = 30

cam = cv2.VideoCapture(0)

def capture_frame() -> None:
    """
    Capture frame and display in ui page.

    Returns
    -------
    None.

    """
    if cam.isOpened():
        read, frame = cam.read()
        if read:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred = get_prediction(frame,model)
            cv2.putText(frame, pred, (10,200),cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 2, cv2.LINE_AA)
            height, width, channel = frame.shape
            step = channel * width
            qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
            page.label.setPixmap(QPixmap.fromImage(qImg))

def start_camera() -> None:
    page.timer.start(round(frame_rate/1000))

def stop_camera() -> None:
    page.timer.stop()


app = QtWidgets.QApplication([])

page = uic.loadUi('page.ui')

page.timer = QTimer()
page.timer.timeout.connect(capture_frame)
page.startbut.clicked.connect(start_camera)
page.stopbut.clicked.connect(stop_camera)

page.show()

app.exec()
app.quit()

cam.release()

del app






