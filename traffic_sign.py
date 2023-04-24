import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

model = load_model('project.h5')
CLASS_NAMES = ['Speed limit (20km/h)',
               'Speed limit (30km/h)',
               'Speed limit (50km/h)',
               'Speed limit (60km/h)',
               'Speed limit (70km/h)',
               'Speed limit (80km/h)',
               'End of speed limit (80km/h)',
               'Speed limit (100km/h)',
               'Speed limit (120km/h)',
               'No passing',
               'No passing veh over 3.5 tons',
               'Right-of-way at intersection',
               'Priority road',
               'Yield',
               'Stop',
               'No vehicles',
               'Veh > 3.5 tons prohibited',
               'No entry',
               'General caution',
               'Dangerous curve left',
               'Dangerous curve right',
               'Double curve',
               'Bumpy road',
               'Slippery road',
               'Road narrows on the right',
               'Road work',
               'Traffic signals',
               'Pedestrians',
               'Children crossing',
               'Bicycles crossing',
               'Beware of ice/snow',
               'Wild animals crossing',
               'End speed + passing limits',
               'Turn right ahead',
               'Turn left ahead',
               'Ahead only',
               'Go straight or right',
               'Go straight or left',
               'Keep right',
               'Keep left',
               'Roundabout mandatory',
               'End of no passing',
               'End no passing veh > 3.5 tons']

st.title('Traffic Sign Classification')
st.markdown('Upload Image')


def numbers_to_strings(argument):
    switcher = {
        0: 'Speed limit (20km/h)',
        1: 'Speed limit (30km/h)',
        2: 'Speed limit (50km/h)',
        3: 'Speed limit (60km/h)',
        4: 'Speed limit (70km/h)',
        5: 'Speed limit (80km/h)',
        6: 'End of speed limit (80km/h)',
        7: 'Speed limit (100km/h)',
        8: 'Speed limit (120km/h)',
        9: 'No passing',
        10: 'No passing veh over 3.5 tons',
        11: 'Right-of-way at intersection',
        12: 'Priority road',
        13: 'Yield',
        14: 'Stop',
        15: 'No vehicles',
        16: 'Veh > 3.5 tons prohibited',
        17: 'No entry',
        18: 'General caution',
        19: 'Dangerous curve left',
        20: 'Dangerous curve right',
        21: 'Double curve',
        22: 'Bumpy road',
        23: 'Slippery road',
        24: 'Road narrows on the right',
        25: 'Road work',
        26: 'Traffic signals',
        27: 'Pedestrians',
        28: 'Children crossing',
        29: 'Bicycles crossing',
        30: 'Beware of ice/snow',
            31: 'Wild animals crossing',
            32: 'End speed + passing limits',
            33: 'Turn right ahead',
            34: 'Turn left ahead',
            35: 'Ahead only',
            36: 'Go straight or right',
            37: 'Go straight or left',
            38: 'Keep right',
            39: 'Keep left',
            40: 'Roundabout mandatory',
            41: 'End of no passing',
        42: 'End no passing veh > 3.5 tons'
    }
    return switcher.get(argument, "nothing")


dog_image = st.file_uploader("Upload image")
submit = st.button('Predict')
if submit:
    if dog_image is not None:
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels="BGR")
        opencv_image = cv2.resize(opencv_image, (30, 30))
        opencv_image.shape = (1, 30, 30, 3)
        Y_pred = model.predict(opencv_image)
        ypred1 = np.argmax(Y_pred)
        predict = ""
        predict = numbers_to_strings(ypred1)
        st.title(predict)
