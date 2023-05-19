import os
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from ctypes import *
from typing import cast

# for yolov5 and fps counter
import torch
import time


sys.path.append(os.path.join(os.path.dirname(__file__), 'camera_sdk'))
from MvCameraControl_class import *
from CameraParams_const import *
from MvErrorDefine_const import *
from CameraParams_header import *

import numpy as np
from threading import Thread, Lock, Event
import cv2

import re

WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 900

IMAGE_WINDOW_SIZE = 640

numpy_image = np.zeros([3648, 5472, 1], "uint8")
camera = MvCamera()

thread_event = Event()



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.weight_path = os.path.join(os.path.dirname(__file__), 'weights/best.pt')

        self.init_yolov5()
        self.setWindowTitle("GUI")
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)

        self.isOPen = False
        self.isGrabbing = False
        self.NLM_flag = False
        self.yolov5_flag = False

        # custom font for setting labels
        self.label_font = QFont("Arial", 12, QFont.Bold)

        # NLM filters parameters
        self.NLM_filter_strength = 3
        self.NLM_search_window = 21
        self.NLM_block_size = 7



        self.UI_components()
        self.object_connections()

    def init_yolov5(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weight_path, force_reload=True)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)

    def apply_yolov5(self, image):
        results = self.model(image)

        labels, coordinates, confidence = results.pandas().xyxy[0].to_numpy()[:, -1], results.pandas().xyxy[0].to_numpy()[:, :4], results.pandas().xyxy[0].to_numpy()[:, -3]

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for i in range(len(labels)):
            row = coordinates[i]
            print(row)
            if confidence[i] > int(self.yolov5_confidence_spinbox.value()) / 100:
                x1, y1, x2, y2 = int(row[0]), \
                                 int(row[1]), \
                                 int(row[2]), \
                                 int(row[3])


                image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image



    def UI_components(self):

        self.image_window = QLabel(self)
        self.image_window.setFixedSize(IMAGE_WINDOW_SIZE, IMAGE_WINDOW_SIZE)
        self.image_window.move(50, int((WINDOW_HEIGHT - IMAGE_WINDOW_SIZE) / 2))
        self.image_window.setStyleSheet('border: 2px solid black;')
        image = QPixmap(os.path.join(os.path.dirname(__file__), 'icons/no_camera.png'))
        self.image_window.setPixmap(image)
        self.camera_settings_UI()
        self.chose_camera_UI()
        self.change_resolution_UI()
        self.NLM_UI()
        self.yolov5_ui()

    def camera_settings_UI(self):
        # Camera settings block
        self.settings_field = QLabel(self)
        self.settings_field.setFixedSize(320, 186)
        self.settings_field.move(self.image_window.width() + 100, self.image_window.y() + 140 + 45)
        self.settings_field.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.settings_field.setLineWidth(1)


        self.setting_labels = [QLabel("Exposure", self.settings_field), QLabel("Gain", self.settings_field), QLabel("Framerate", self.settings_field)]
        for label in self.setting_labels:
            label.setFont(self.label_font)
            label.adjustSize()

        # place labels inside the field
        self.setting_labels[0].move(30, 15)
        self.setting_labels[1].move(30, self.setting_labels[0].y() + self.setting_labels[0].height() + 20)
        self.setting_labels[2].move(30, self.setting_labels[1].y() + self.setting_labels[1].height() + 20)


        self.setting_inputs = [QLineEdit(self.settings_field), QLineEdit(self.settings_field), QLineEdit(self.settings_field)]
        for input in self.setting_inputs:
            input.setText("0")
            input.setFixedSize(120, self.setting_labels[0].height())

        # place inputs successively with setting labels
        self.setting_inputs[0].move(int((self.settings_field.width() / 2) + 10), self.setting_labels[0].y())
        self.setting_inputs[1].move(int((self.settings_field.width() / 2) + 10), self.setting_labels[1].y())
        self.setting_inputs[2].move(int((self.settings_field.width() / 2) + 10), self.setting_labels[2].y())


        self.get_parameters_button = QPushButton("Get parameters", self.settings_field)
        self.set_parameters_button = QPushButton("Set parameters", self.settings_field)

        self.get_parameters_button.setFixedSize(self.setting_inputs[0].width(), self.setting_labels[0].height())
        self.set_parameters_button.setFixedSize(self.setting_inputs[0].width(), self.setting_labels[0].height())

        self.get_parameters_button.move(30, self.setting_labels[2].y() + self.setting_labels[2].height() + 20)

        self.set_parameters_button.move(self.setting_inputs[2].x(), self.setting_labels[2].y() + self.setting_labels[2].height() + 20)
        self.get_parameters_button.setEnabled(False)
        self.set_parameters_button.setEnabled(False)

    def chose_camera_UI(self):
        self.cameras_list = QComboBox(self)
        self.cameras_list.setFixedSize(550, 40)
        self.cameras_list.move(self.image_window.x() + 50, 60)


        # chose camera block
        self.choice_field = QLabel(self)
        self.choice_field.setFixedSize(320, 140 + 45)
        self.choice_field.move(self.image_window.width() + 100, self.image_window.y())
        self.choice_field.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.choice_field.setLineWidth(1)

        self.check_cameras = QPushButton("Check cameras", self.choice_field)
        self.check_cameras.move(30, 15)
        self.check_cameras.setFixedSize(260, self.get_parameters_button.height())

        self.open_button = QPushButton("Open camera", self.choice_field)
        self.close_button = QPushButton("Close camera", self.choice_field)

        self.open_button.setFixedSize(self.get_parameters_button.size())
        self.close_button.setFixedSize(self.set_parameters_button.size())

        self.open_button.move(self.get_parameters_button.x(), self.check_cameras.y() + self.check_cameras.height() + 20)
        self.close_button.move(self.set_parameters_button.x(), self.check_cameras.y() + self.check_cameras.height() + 20)

        self.start_grabbing_button = QPushButton("Start Grabbing", self.choice_field)
        self.stop_grabbing_button = QPushButton("Stop Grabbing", self.choice_field)
        self.take_photo_button = QPushButton("Take a photo", self.choice_field)


        self.start_grabbing_button.setFixedSize(self.get_parameters_button.size())
        self.stop_grabbing_button.setFixedSize(self.set_parameters_button.size())
        self.take_photo_button.setFixedSize(self.get_parameters_button.width() * 2 + 20, self.get_parameters_button.height())


        self.start_grabbing_button.move(self.open_button.x(), self.open_button.y() + self.open_button.height() + 20)
        self.stop_grabbing_button.move(self.close_button.x(),
                               self.close_button.y() + self.close_button.height() + 20)

        self.take_photo_button.move(self.open_button.x(),  self.start_grabbing_button.y() + self.start_grabbing_button.height() + 20)

        self.open_button.setEnabled(False)
        self.close_button.setEnabled(False)
        self.start_grabbing_button.setEnabled(False)
        self.stop_grabbing_button.setEnabled(False)
        self.take_photo_button.setEnabled(False)

    def change_resolution_UI(self):
        resolution_field = QLabel(self)
        resolution_field.setFixedSize(320, 175)
        resolution_field.move(self.settings_field.x(), self.settings_field.y() + self.settings_field.height())
        resolution_field.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        resolution_field.setLineWidth(1)

        self.dimension_labels = [QLabel("Width", resolution_field), QLabel("Height", resolution_field)]
        for label in self.dimension_labels:
            label.setFont(self.label_font)
            label.adjustSize()

        self.dimension_inputs = [QLineEdit(resolution_field), QLineEdit(resolution_field)]
        for input in self.dimension_inputs:
            input.setText("0")
            input.setFixedSize(120, self.setting_labels[0].height())

        self.dimension_labels[0].move(30, 15)
        self.dimension_labels[1].move(30, self.dimension_labels[0].y() + self.dimension_labels[0].height() + 20)

        self.dimension_inputs[0].move(int((resolution_field.width() / 2) + 10), self.dimension_labels[0].y())
        self.dimension_inputs[1].move(int((resolution_field.width() / 2) + 10), self.dimension_labels[1].y())

        self.get_resolution_button = QPushButton("Get resolution", resolution_field)
        self.set_resolution_button = QPushButton("Set resolution", resolution_field)
        self.set_default_resolution_button = QPushButton("Set default resolution", resolution_field)

        self.get_resolution_button.setFixedSize(self.get_parameters_button.size())
        self.set_resolution_button.setFixedSize(self.set_parameters_button.size())
        self.set_default_resolution_button.setFixedSize(self.set_parameters_button.width() * 2 + 20, self.set_parameters_button.height())


        self.get_resolution_button.move(30, self.dimension_labels[1].y() + self.dimension_labels[1].height() + 20)

        self.set_resolution_button.move(self.dimension_inputs[1].x(),
                                        self.dimension_inputs[1].y() + self.dimension_labels[1].height() + 20)

        self.set_default_resolution_button.move(self.get_resolution_button.x(),
                                                self.get_resolution_button.y() + self.get_resolution_button.height() + 10)

        self.get_resolution_button.setEnabled(False)
        self.set_resolution_button.setEnabled(False)
        self.set_default_resolution_button.setEnabled(False)

    def NLM_UI(self):
        # create a field for NLM adjustment parameters
        self.NLM_field = QLabel(self)
        self.NLM_field.setFixedSize(320, 140 + 45)
        self.NLM_field.move(self.settings_field.x() + self.settings_field.width(), self.image_window.y())
        self.NLM_field.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.NLM_field.setLineWidth(1)

        self.NLM_switcher = QPushButton("NLM filter", self.NLM_field)
        self.NLM_switcher.setFixedSize(self.check_cameras.width(), self.check_cameras.height())
        self.NLM_switcher.move(int(self.NLM_field.width() / 2 - self.NLM_switcher.width() / 2), self.check_cameras.y())

        self.NLM_status_label = QLabel('-', self.NLM_field)
        self.NLM_status_label.setFont(self.label_font)
        self.NLM_status_label.adjustSize()
        self.NLM_status_label.move(self.NLM_switcher.x() + self.NLM_switcher.width() + 5, self.NLM_switcher.y() - 2)


        # create text labels
        self.NLM_strength_label = QLabel("Filter strength:", self.NLM_field)
        self.NLM_seacrh_window_label = QLabel("Search window:", self.NLM_field)
        self.NLM_block_size_label = QLabel("Block size:", self.NLM_field)
        # set specific font for them
        self.NLM_strength_label.setFont(self.label_font)
        self.NLM_seacrh_window_label.setFont(self.label_font)
        self.NLM_block_size_label.setFont(self.label_font)
        # adjust size to the text
        self.NLM_strength_label.adjustSize()
        self.NLM_seacrh_window_label.adjustSize()
        self.NLM_block_size_label.adjustSize()
        # create input spinboxes
        self.NLM_strength_spinbox = QSpinBox(self.NLM_field)
        self.NLM_seacrh_window_spinbox = QSpinBox(self.NLM_field)
        self.NLM_block_size_spinbox = QSpinBox(self.NLM_field)
        # set required width to them
        self.NLM_strength_spinbox.setFixedWidth(60)
        self.NLM_seacrh_window_spinbox.setFixedWidth(60)
        self.NLM_block_size_spinbox.setFixedWidth(60)
        # adjust their size to real dimensions
        self.NLM_strength_spinbox.adjustSize()
        self.NLM_seacrh_window_spinbox.adjustSize()
        self.NLM_block_size_spinbox.adjustSize()
        # set actual range for filter parameters
        self.NLM_strength_spinbox.setRange(0, 50)
        self.NLM_seacrh_window_spinbox.setRange(0, 50)
        self.NLM_seacrh_window_spinbox.setRange(0, 50)

        # set initial values into the spinboxes
        self.NLM_strength_spinbox.setValue(self.NLM_filter_strength)
        self.NLM_seacrh_window_spinbox.setValue(self.NLM_search_window)
        self.NLM_block_size_spinbox.setValue(self.NLM_block_size)

        self.NLM_strength_label.move(self.NLM_switcher.x(), self.open_button.y())
        self.NLM_strength_spinbox.move(self.NLM_switcher.x() + self.NLM_switcher.width() - self.NLM_strength_spinbox.width(), self.open_button.y())

        self.NLM_seacrh_window_label.move(self.NLM_switcher.x(), self.start_grabbing_button.y())
        self.NLM_seacrh_window_spinbox.move(self.NLM_switcher.x() + self.NLM_switcher.width() - self.NLM_seacrh_window_spinbox.width(), self.start_grabbing_button.y())

        self.NLM_block_size_label.move(self.NLM_switcher.x(), self.take_photo_button.y())
        self.NLM_block_size_spinbox.move(self.NLM_switcher.x() + self.NLM_switcher.width() - self.NLM_block_size_spinbox.width(), self.take_photo_button.y())

        # initially switch-off button and all input fields
        self.NLM_switcher.setEnabled(False)
        self.NLM_strength_spinbox.setEnabled(False)
        self.NLM_seacrh_window_spinbox.setEnabled(False)
        self.NLM_block_size_spinbox.setEnabled(False)

    def yolov5_ui(self):
        # create a field for NLM adjustment parameters
        self.yolov5_field = QLabel(self)
        self.yolov5_field.setFixedSize(320, 140 + 45)
        self.yolov5_field.move(self.settings_field.x() + self.settings_field.width(), self.settings_field.y())
        self.yolov5_field.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)
        self.yolov5_field.setLineWidth(1)

        # create a button to switch-on (off) yolov5
        self.yolov5_switcher = QPushButton('YOLOv5', self.yolov5_field)
        self.yolov5_switcher.setFixedSize(self.check_cameras.width(), self.check_cameras.height())
        self.yolov5_switcher.move(int(self.yolov5_field.width() / 2 - self.yolov5_switcher.width() / 2), self.setting_labels[0].y())

        self.yolov5_status_label = QLabel('-', self.yolov5_field)
        self.yolov5_status_label.setFont(self.label_font)
        self.yolov5_status_label.adjustSize()
        self.yolov5_status_label.move(self.yolov5_switcher.x() + self.yolov5_switcher.width() + 5, self.yolov5_switcher.y() - 2)

        # label for confidence
        self.yolov5_confidence_label = QLabel('Confidence', self.yolov5_field)
        # set up the font
        self.yolov5_confidence_label.setFont(self.label_font)
        # adjust size to the text
        self.yolov5_confidence_label.adjustSize()

        # create input spinbox
        self.yolov5_confidence_spinbox = QSpinBox(self.yolov5_field)
        # set required width to them
        self.yolov5_confidence_spinbox.setFixedWidth(60)
         # set actual range for filter parameters
        self.yolov5_confidence_spinbox.setRange(0, 100)
        # set initial values into the spinboxes
        self.yolov5_confidence_spinbox.setValue(50)

        self.yolov5_confidence_label.move(self.setting_labels[1].x(), self.setting_labels[1].y())
        self.yolov5_confidence_spinbox.move(self.yolov5_switcher.x() + self.yolov5_switcher.width() - self.yolov5_confidence_spinbox.width(), self.setting_labels[1].y())

        self.yolov5_switcher.setEnabled(False)
        self.yolov5_confidence_spinbox.setEnabled(False)


    def find_cameras(self):
        camera = MvCamera()
        # next we need to find all cameras connected to GIGE port
        self.available_cameras = MV_CC_DEVICE_INFO_LIST()
        ret = camera.MV_CC_EnumDevices(MV_GIGE_DEVICE, self.available_cameras)
        if self.available_cameras.nDeviceNum == 0:
            print('No camera detected!')
        else:
            self.open_button.setEnabled(True)
            print(f'Find {self.available_cameras.nDeviceNum} devices!')
            for i in range(self.available_cameras.nDeviceNum):
                # Override type checker to think that device_info has POINTER(MV_CC_DEVICE_INFO) type
                # .contents - get an object from pointer
                device_info = cast(self.available_cameras.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
                if device_info.nTLayerType == MV_GIGE_DEVICE:

                    camera_manufacturer_name = ""
                    for ch in device_info.SpecialInfo.stGigEInfo.chModelName:
                        if ch == 0:
                            break
                        camera_manufacturer_name += chr(ch)

                    # read ip address byte by byte (only in order to print it in beautiful form)
                    camera_ip1 = ((device_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                    camera_ip2 = ((device_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                    camera_ip3 = ((device_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                    camera_ip4 = (device_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                    ip_address = f"{camera_ip1}.{camera_ip2}.{camera_ip3}.{camera_ip4}"
                    self.cameras_list.addItem(f"{camera_manufacturer_name}: {ip_address}")


    def open_camera(self):
        # check if the device is accessible
        # 0 - something went wrong
        # 1 - device is accessible
        # get device info and cast it to MV_CC_DEFICE_INFO pointer type
        self.found_camera = cast(self.available_cameras.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = camera.MV_CC_IsDeviceAccessible(self.found_camera, MV_ACCESS_Exclusive)
        if ret:
            # create a handler to work with a camera
            ret = camera.MV_CC_CreateHandle(self.found_camera)
            if ret == MV_OK:
                # try to connect to the found camera
                # function's default parameters
                # unsigned int nAccessMode = MV_ACCESS_Exclusive,
                # unsigned short nSwitchoverKey = 0
                ret = camera.MV_CC_OpenDevice()
                if ret == MV_OK:
                    self.isOPen = True
                    self.open_button.setEnabled(False)
                    self.close_button.setEnabled(True)
                    self.start_grabbing_button.setEnabled(True)
                    self.get_resolution_button.setEnabled(True)
                    self.set_resolution_button.setEnabled(True)
                    self.set_default_resolution_button.setEnabled(True)
                    self.get_parameters_button.setEnabled(True)
                    self.set_parameters_button.setEnabled(True)

                    self.set_resolution()
                    self.get_parameters()
                    optimal_package_size = camera.MV_CC_GetOptimalPacketSize()
                    print(f'Optimal package size: {optimal_package_size}')
                    if optimal_package_size > 0:
                        # "GevSCPSPacketSize"
                        # Specifies the size of a stream packet that is sent on the stream channel. The unit of the packet size is bytes.
                        camera.MV_CC_SetIntValue("GevSCPSPacketSize", optimal_package_size)
                        # Set the trigger mode to off
                        camera.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
                        # Get the enumerator name according to the node name and assigned value
                        stEnumValue = MVCC_ENUMVALUE()
                        memset(byref(stEnumValue), 0, sizeof(MVCC_ENUMVALUE))
                        camera.MV_CC_GetEnumValue("PixelFormat", stEnumValue)
                        print(f'Pixel format: {stEnumValue.nCurValue} (need to compare with a table)')



    def close_camera(self):
        self.close_button.setEnabled(False)
        self.open_button.setEnabled(True)
        self.start_grabbing_button.setEnabled(False)
        self.get_resolution_button.setEnabled(False)
        self.set_resolution_button.setEnabled(False)
        self.set_default_resolution_button.setEnabled(False)
        self.get_parameters_button.setEnabled(False)
        self.set_parameters_button.setEnabled(False)

        # After connecting to device via calling API MV_CC_OpenDevice , you can call this API to disconnect and release resources.
        camera.MV_CC_CloseDevice()
        # Destroy camera handler
        camera.MV_CC_DestroyHandle()

    def work_thread(self):
        global numpy_image, camera
        # get the total size of a frame in pixels (for example  5472 × 3648 = 19961856)
        image_size = MVCC_INTVALUE_EX()
        ret = camera.MV_CC_GetIntValueEx("PayloadSize", image_size)

        # structure to get all info about the frame (size, RGB, frame number ...)
        frame_info = MV_FRAME_OUT_INFO_EX()
        # print(f'Image size in pixels: {image_size.nCurValue}')

        if ret == MV_OK:
            need_buffer_size = int(image_size.nCurValue)
        else:
            return
        # buffer to save pixels of the receiving frame
        buffer_size = 0
        buffer = 0
        while thread_event.is_set():
            if buffer_size < need_buffer_size:
                # create an array with the buffer size, initialized with 0
                buffer = (c_ubyte * need_buffer_size)()
                buffer_size = need_buffer_size

            start_time = time.perf_counter()

            # get a data from the camera in the buffer
            ret = camera.MV_CC_GetOneFrameTimeout(buffer, buffer_size, frame_info)
            # print("Received a frame: Width[%d], Height[%d], nFrameNum[%d]"
            #       % (frame_info.nWidth, frame_info.nHeight, frame_info.nFrameNum))






            numpy_image = self.convert_buffer_image(buffer, frame_info.nWidth, frame_info.nHeight)
            if frame_info.nWidth != 0:
                image = cv2.resize(numpy_image, (IMAGE_WINDOW_SIZE, IMAGE_WINDOW_SIZE), interpolation=cv2.INTER_AREA)

                dim = (640, int(numpy_image.shape[0] * (640 / numpy_image.shape[1])))
                resized_img = cv2.resize(numpy_image, dim, cv2.INTER_AREA)
                if self.NLM_flag:
                    resized_img = cv2.fastNlMeansDenoising(resized_img, None, self.NLM_filter_strength, self.NLM_block_size, self.NLM_search_window)


                padded_img = np.zeros((IMAGE_WINDOW_SIZE, IMAGE_WINDOW_SIZE), dtype=np.uint8)

                # center an image
                start_pixel = int((padded_img.shape[0] - resized_img.shape[0]) / 2)
                end_pixel = start_pixel + resized_img.shape[0]
                # do padding
                for h in range(padded_img.shape[0]):
                    for w in range(padded_img.shape[1]):
                        if start_pixel <= h < end_pixel:
                            padded_img[h, w] = resized_img[h - start_pixel, w]


                # display a padded image
                if self.yolov5_flag:
                    padded_img = self.apply_yolov5(padded_img)

                self.image_window.setPixmap(self.convert_cv_qt(padded_img))

                end_time = time.perf_counter()
                fps = 1 / np.round(end_time - start_time, 3)
                # print(fps)


    def convert_buffer_image(self, buffer, width, height):
        data = np.frombuffer(buffer, count=int(width * height), dtype=np.uint8, offset=0)
        data = data.reshape(height, width)
        image_np = np.zeros([height, width, 1], "uint8")
        image_np[:, :, 0] = data

        return image_np

    def start_grabbing_images(self):
        self.start_grabbing_button.setEnabled(False)
        self.stop_grabbing_button.setEnabled(True)
        self.close_button.setEnabled(False)

        self.get_resolution_button.setEnabled(False)
        self.set_resolution_button.setEnabled(False)
        self.set_default_resolution_button.setEnabled(False)
        self.take_photo_button.setEnabled(True)
        self.get_parameters_button.setEnabled(False)
        self.set_parameters_button.setEnabled(False)

        self.NLM_switcher.setEnabled(True)
        self.NLM_strength_spinbox.setEnabled(True)
        self.NLM_seacrh_window_spinbox.setEnabled(True)
        self.NLM_block_size_spinbox.setEnabled(True)

        self.yolov5_switcher.setEnabled(True)
        self.yolov5_confidence_spinbox.setEnabled(True)


        ret = camera.MV_CC_StartGrabbing()
        thread_event.set()
        self.thread1 = Thread(target=self.work_thread, args=())
        self.thread1.start()

    def stop_grabbing_images(self):

        self.start_grabbing_button.setEnabled(True)
        self.stop_grabbing_button.setEnabled(False)
        self.close_button.setEnabled(True)

        self.get_resolution_button.setEnabled(True)
        self.set_resolution_button.setEnabled(True)
        self.set_default_resolution_button.setEnabled(True)
        self.take_photo_button.setEnabled(False)
        self.get_parameters_button.setEnabled(True)
        self.set_parameters_button.setEnabled(True)

        self.NLM_switcher.setEnabled(False)
        self.NLM_strength_spinbox.setEnabled(False)
        self.NLM_seacrh_window_spinbox.setEnabled(False)
        self.NLM_block_size_spinbox.setEnabled(False)

        self.yolov5_switcher.setEnabled(False)
        self.yolov5_confidence_spinbox.setEnabled(False)


        self.NLM_flag = False
        self.NLM_status_label.setText('-')
        self.NLM_status_label.adjustSize()
        self.NLM_status_label.move(self.NLM_switcher.x() + self.NLM_switcher.width() + 5, self.NLM_switcher.y() - 2)

        self.yolov5_flag = False
        self.yolov5_status_label.setText('-')
        self.yolov5_status_label.adjustSize()
        self.yolov5_status_label.move(self.yolov5_switcher.x() + self.yolov5_switcher.width() + 5, self.yolov5_switcher.y() - 2)

        ret = camera.MV_CC_StopGrabbing()
        thread_event.clear()


    def convert_cv_qt(self, img):
        # rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img.shape)
        if len(img.shape) == 2:
            h, w = img.shape
            qImg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
            p = qImg.scaled(640, 640, Qt.KeepAspectRatio)
            return QPixmap.fromImage(p)
        elif len(img.shape) == 3:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(640, 640, Qt.KeepAspectRatio)
            return QPixmap.fromImage(p)


    def convert_cv_qt_color(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 640, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)



    def get_resolution(self):
        height = MVCC_INTVALUE()
        memset(byref(height), 0, sizeof(MVCC_INTVALUE))
        camera.MV_CC_GetIntValue("Height", height)

        width = MVCC_INTVALUE()
        memset(byref(width), 0, sizeof(MVCC_INTVALUE))
        camera.MV_CC_GetIntValue("Width", width)

        self.dimension_inputs[0].setText(str(width.nCurValue))
        self.dimension_inputs[1].setText(str(height.nCurValue))


    def set_resolution(self):
        match_width = re.search(r"^\d{1,4}$", self.dimension_inputs[0].text())
        match_height = re.search(r"^\d{1,4}$", self.dimension_inputs[1].text())

        if match_width and match_height:
            camera.MV_CC_SetIntValue("Width", int(match_width.group()))
            camera.MV_CC_SetIntValue("Height", int(match_height.group()))

            self.get_resolution()
        else:
            self.dimension_inputs[0].setText("Wrong data")
            self.dimension_inputs[1].setText("Wrong data")

    def get_parameters(self):
        framerate = MVCC_FLOATVALUE()
        memset(byref(framerate), 0, sizeof(MVCC_FLOATVALUE))
        exposure = MVCC_FLOATVALUE()
        memset(byref(exposure), 0, sizeof(MVCC_FLOATVALUE))
        gain = MVCC_FLOATVALUE()
        memset(byref(gain), 0, sizeof(MVCC_FLOATVALUE))

        camera.MV_CC_GetFloatValue("AcquisitionFrameRate", framerate)
        camera.MV_CC_GetFloatValue("ExposureTime", exposure)
        camera.MV_CC_GetFloatValue("Gain", gain)
        self.setting_inputs[0].setText(str(round(exposure.fCurValue, 2)))
        self.setting_inputs[1].setText(str(round(gain.fCurValue, 2)))
        self.setting_inputs[2].setText(str(round(framerate.fCurValue, 2)))

    def set_parameters(self):
        exposure = re.search(r"^\d{1,8}(\.\d+)?$", self.setting_inputs[0].text())
        gain = re.search(r"^\d{1,4}(\.\d+)?$", self.setting_inputs[1].text())
        framerate = re.search(r"^\d{1,4}(\.\d+)?$", self.setting_inputs[2].text())

        camera.MV_CC_SetFloatValue("AcquisitionFrameRate", float(framerate.group()))
        camera.MV_CC_SetFloatValue("ExposureTime", float(exposure.group()))
        camera.MV_CC_SetFloatValue("Gain", float(gain.group()))
        self.get_parameters()

    def set_default_resolution(self):
        camera.MV_CC_SetIntValue("Width", 5472)
        camera.MV_CC_SetIntValue("Height", 3648)
        self.get_resolution()

    def save_photo(self):
         cwd = os.getcwd()
         try:
             os.mkdir(os.path.join(cwd, "images"))
         except FileExistsError:
             cwd = os.path.join(cwd, "images")
             images_folder = os.listdir(cwd)
             num = - 1
             max = 0
             print(sorted(images_folder))
             if len(images_folder) > 0:
                 for d in sorted(images_folder):
                     if re.search(r'^i\D+(\d+)\.png$', d):
                         num = int(re.search(r'^i\D+(\d+)\.png$', d).groups()[0])
                         if num > max:
                             max = num
                             print(max)

                 # if there are files with a different name that doesn't comply with the rules of a regex
                 if num != -1:
                     num += 1
                 else:
                     num = 0
             else:
                 num = 0
             max += 1
             cv2.imwrite(os.path.join(cwd, 'img_' + str(max) + '.png'), numpy_image)
         else:
             cwd = os.path.join(cwd, "../camera_SDK/images")
             num = 0
             cv2.imwrite(os.path.join(cwd, 'img_' + str(num) + '.png'), numpy_image)

    def adjust_NLM(self, parameter):
        if parameter == 'strength':
            self.NLM_filter_strength = int(self.NLM_strength_spinbox.value())
        elif parameter == 'window':
            self.NLM_search_window = int(self.NLM_seacrh_window_spinbox.value())
        elif parameter == 'block':
            self.NLM_block_size = int(self.NLM_block_size_spinbox.value())

    def change_NLM_flag(self):
        self.NLM_flag = not self.NLM_flag
        if self.NLM_flag == True:
            self.NLM_status_label.setText('+')
            self.NLM_status_label.adjustSize()
            self.NLM_status_label.move(self.NLM_switcher.x() + self.NLM_switcher.width() + 5, self.NLM_switcher.y())

        else:
            self.NLM_status_label.setText('-')
            self.NLM_status_label.adjustSize()
            self.NLM_status_label.move(self.NLM_switcher.x() + self.NLM_switcher.width() + 5, self.NLM_switcher.y() - 2)

    def change_yolov5_flag(self):
        self.yolov5_flag = not self.yolov5_flag

        if self.yolov5_flag == True:
            self.yolov5_status_label.setText('+')
            self.yolov5_status_label.adjustSize()
            self.yolov5_status_label.move(self.yolov5_switcher.x() + self.yolov5_switcher.width() + 5, self.yolov5_switcher.y())

        else:
            self.yolov5_status_label.setText('-')
            self.yolov5_status_label.adjustSize()
            self.yolov5_status_label.move(self.yolov5_switcher.x() + self.yolov5_switcher.width() + 5, self.yolov5_switcher.y() - 2)





    # I want to create Real time defect detection software
    # YOLOv5 need 1500 images minimum for good accuracy
    def object_connections(self):
        self.check_cameras.clicked.connect(self.find_cameras)
        self.open_button.clicked.connect(self.open_camera)
        self.close_button.clicked.connect(self.close_camera)
        self.start_grabbing_button.clicked.connect(self.start_grabbing_images)
        self.stop_grabbing_button.clicked.connect(self.stop_grabbing_images)
        self.get_resolution_button.clicked.connect(self.get_resolution)
        self.set_resolution_button.clicked.connect(self.set_resolution)
        self.set_default_resolution_button.clicked.connect(self.set_default_resolution)
        self.take_photo_button.clicked.connect(self.save_photo)
        self.get_parameters_button.clicked.connect(self.get_parameters)
        self.set_parameters_button.clicked.connect(self.set_parameters)
        self.NLM_strength_spinbox.valueChanged.connect(lambda: self.adjust_NLM('strength'))
        self.NLM_seacrh_window_spinbox.valueChanged.connect(lambda: self.adjust_NLM('window'))
        self.NLM_block_size_spinbox.valueChanged.connect(lambda: self.adjust_NLM('block'))
        self.NLM_switcher.clicked.connect(self.change_NLM_flag)
        self.yolov5_switcher.clicked.connect(self.change_yolov5_flag)

    def closeEvent(self, event):
        thread_event.clear()
        event.accept()

app = QApplication(sys.argv)

main_window = MainWindow()
main_window.show()

app.exec_()

