import cv2
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage

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
page.pushButton.clicked.connect(start_camera)
page.pushButton_2.clicked.connect(stop_camera)

page.show()

app.exec()
app.quit()

cam.release()

del app
