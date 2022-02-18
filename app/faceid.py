# Import dependencies
from turtle import onclick
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy depencies
from kivy.clock import Clock 
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from layers import L1Dist


class CamApp(App):

    def build(self):
        # Setting up our layout components
        self.webcam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify",on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text="Verification Off", size_hint=(1,.1))

        #adding variables to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.webcam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        #loading our model
        self.model = load_model('siamesemodelv3.h5', custom_objects={'L1Dist':L1Dist})

        #Setting up cv2 videocapture
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.updateCamera, 1.0/33.0)

        return layout

    #Run continuously to get webcam feed
    def updateCamera(self, *args):
        #Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 220:220+250, :]

        #flip horizontal and convert image to texture
        buff = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buff, colorfmt='bgr', bufferfmt='ubyte')
        self.webcam.texture = img_texture

    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # load images
        img = tf.io.decode_jpeg(byte_img)
        # Preprocessing images to 105 to 105 pixels
        img = tf.image.resize(img, (105, 105))
        # Scale image to be between 0 and 1
        img = img / 255.0
        return img

    def verify(self,*args):
        #Specifying threshold
        detection_threshold = .5
        verification_threshold = .5

        #Saving our input image
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 220:220+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        #Getting access to our Paths both input_image/validation_image
        #Precocessing images in both directory
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        # Detection Threshold: Metric above which a prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)
        
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        self.verification_label.text = 'Verified' if verified == True else 'Unverified'
        
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        return results, verified


if __name__ == '__main__':
    CamApp().run()