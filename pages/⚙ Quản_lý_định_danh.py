# from mtcnn import MTCNN
import os.path
import time
from datetime import date

import numpy as np
import streamlit as st
import cv2
import pandas as pd
import glob
# import shutil
# import pyodbc
import requests
import base64

st.set_page_config(layout='wide')
st.title('Quản lý thông tin định danh sinh viên')
st.write('Hệ thống cung cấp công cụ quản lý thông tin đăng nhập sinh viên hiệu quả')

tab_add, tab_modify, tab_remove = st.tabs(['Thêm thông tin định danh', 'Sửa thông tin định danh', 'Xoá thông tin định danh'])

def load_images(folder_name, file_extension):
    image_files = glob.glob(f'{folder_name}/*/*.{file_extension}')
    manuscripts = []
    for image_file in image_files:
        image_file = image_file.replace('\\', '/')
        parts = image_file.split('/')
        if parts[1] not in manuscripts:
            manuscripts.append(parts[1])

    return image_files, manuscripts

def face_detector(face_cascade_pretrain, frame, id, fullname):
    frame_copy = frame.copy()
    gray_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_pretrain.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (255, 255, 0), 2)

    return frame_copy

def auto_capture_save_images(output_folder, total_pictures_limit):
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    cap = cv2.VideoCapture(0)

    capture_time_ms = 500
    image_count = 1                                                                                          # Đặt biến đếm để đánh số ảnh
    timer = time.time() + (capture_time_ms / 1000)                                                           # Hẹn giờ để chụp ảnh mỗi giây
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')      # Tải pretrain classifier để phát hiện khuôn mặt
    while True:
        ret, frame = cap.read()
        frame_copy = frame.copy()                                                                            # Tạo một bản sao của khung hình gốc để vẽ bounding box và hiển thị
        gray_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Auto capture image for training', frame_copy)                                            # Hiển thị khung hình bản sao có bounding box

        if time.time() >= timer and len(faces) > 0:                                                          # Kiểm tra nếu đến thời gian chụp ảnh và phát hiện được khuôn mặt
            image_name = f"{output_folder}/{image_count}.jpg"                                                # Tạo tên file ảnh dựa trên thời gian hiện tại
            
            
            # Convert frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer)

            # Send image to Node.js server via API
            # url = 'http://14.251.76.129:3107/upload'
            url = 'http://1.55.90.150:3107/upload'
            headers = {'Content-Type': 'application/json'}
            data = {'image': frame_base64.decode('utf-8'), 'name': output_folder, 'image_count': image_count}
            response = requests.post(url, json=data, headers=headers)

            print(response)
            
            
            # cv2.imwrite(image_name, frame)                                                                   # Lưu ảnh từ khung hình gốc vào thư mục đích
            print(f"Saved image: {image_name}")
            image_count += 1

            if image_count > total_pictures_limit:
                break
            timer = time.time() + (capture_time_ms / 1000)                                                   # Cập nhật thời gian chụp ảnh tiếp theo

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return image_count - 1

with tab_add:
    col_1, col_2 = st.columns([5, 2.6])
    with col_1:
        id = st.text_input('Mã sinh viên', placeholder='Mã sinh viên')
        fullName = str(st.text_input('Họ và tên của sinh viên', placeholder='Họ và tên sinh viên'))

        upload_type = st.radio('Vui lòng lựa chọn hình thức tải dữ liệu', ('Tải ảnh lên', 'Sử dụng camera'))
        if upload_type == 'Tải ảnh lên':
            uploaded_images = st.file_uploader('Tải ảnh lên tại đây', type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
            if uploaded_images is not None:
                n_add_view = 10
                groups = []
                for i in range(0, len(uploaded_images), n_add_view):
                    groups.append(uploaded_images[i:i + n_add_view])

                for group in groups:
                    cols = st.columns(n_add_view)
                    for i, image_file in enumerate(group):
                        cols[i].image(image_file)
        else:
            if id != '' and fullName != '':
                destination_folder_path = f'./train_img/{id} - {fullName}'
                total_image = auto_capture_save_images(destination_folder_path, 200)
                st.success(f'Thêm dữ liệu định danh của sinh viên **{fullName}** vào CSDL thành công')

