from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import cv2

import numpy as np
import tensorflow as tf
from scipy import misc

import facenet_tf.src.align.detect_face as detect_face
from facenet_tf import init_value
from facenet_tf.src.common import facenet
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib.pyplot as plt
import datetime
import logging
import dlib
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner
import facenet_tf.src.common.utils as utils
import face_recognition
import facenet_tf.src.common.predict as predict

class FaceRecognitionRun():
    def _init_value(self, runtype):
        self.runtype = runtype # test, real
        self.readImageSizeX = 0.5
        self.readImageSizeY = 0.5
        self.viewImageSizeX = 2
        self.viewImageSizeY = 2

        self.findlist = ['', '', '']  # 배열에 모두 동일한 값이 들어가야 인증이 됨.
        self.boxes_min = 50  # detect min box size
        self.stand_box = [30, 20]  # top left(width, height)
        self.prediction_max = 0.05  # 이 수치 이상 정합성을 보여야 인정 됨.
        self.prediction_svm_pair_max = 0.501
        self.prediction_sub_max = 0.0001
        self.prediction_cos_max = 0.0001
        self.prediction_cnt = 6
        # 로그를 보여주는 개수를 정함
        self.eval_log_cnt = 3000 # 평가중 몇번마다 로그를 찍을지 결정을 한다.

        self.stand_box_color = (255, 255, 255)
        self.alert_color = (0, 0, 255)
        self.box_color = (120, 160, 230)
        self.text_color = (0, 255, 0)

        self.name_font_size = 16
        self.result_font_size = 18
        self.alert_font_size = 18

    def realtime_run(self, runtype = 'real'):
        init_value.init_value.init(self)
        self._init_value(runtype)

        if runtype == 'test': # GPU
            with tf.Graph().as_default():
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
                sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
                with sess.as_default():
                    self.realtime_run_sess(sess)
        else: # CPU
            config = tf.ConfigProto( device_count={'GPU': 0} )
            with tf.Session(config=config) as sess:
                self.realtime_run_sess(sess)

    def realtime_run_sess(self, sess):
        self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)

        # Pre Train Load the model
        print('Loading feature extraction model')
        facenet.load_model(self.feature_model)

        # Get input and output tensors
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # Model Load
        if self.pair_type.count('distance') == 0:
            classifier_filename_exp = os.path.expanduser(self.classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (self.model, self.class_names) = pickle.load(infile)
                print('load classifier file-> %s' % classifier_filename_exp)
                print('')

        self.logger = logging.getLogger('myapp')
        hdlr = logging.FileHandler(self.log_dir + '/myface.log')
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.WARNING)

        # dlib rotation
        self.predictor = dlib.shape_predictor(self.pre_model_dir + self.predictor_68_face_landmarks.replace('.bz2', ''))
        self.detector = dlib.get_frontal_face_detector()
        self.fa = FaceAligner(self.predictor, desiredFaceWidth=self.image_size + self.cropped_size)

        self.gallery_load_flag = True

        if self.runtype == 'test':
            self.debug = True
            predict.getpredict_test(self, sess)
        else:
            self.debug = True
            self.pretime = '99'  # 1 second save
            video_capture = cv2.VideoCapture(0)
            self.predict_flag = True

            self.stand_box_flag = True

            while True:
                ret, frame = video_capture.read()
                frame = cv2.flip(frame, 1)
                self.saveframe = frame
                frame = cv2.resize(frame, (0, 0), fx=self.readImageSizeX, fy=self.readImageSizeY)
                if self.stand_box_flag:
                    self.stand_box = [self.stand_box[0], self.stand_box[1]]
                    self.stand_box.append(frame.shape[1] - self.stand_box[0])
                    self.stand_box.append(frame.shape[0] - self.stand_box[1])

                frame = predict.getpredict(self, sess, frame)

                frame = cv2.resize(frame, (0, 0), fx=self.viewImageSizeX, fy=self.viewImageSizeY)
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            video_capture.release()
            cv2.destroyAllWindows()


    def draw_border(self, img, pt1, pt2, color, thickness, r, d):
        x1, y1 = pt1
        x2, y2 = pt2

        # Top left
        cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

        # Top right
        cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

        # Bottom left
        cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

        # Bottom right
        cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

        return img

if __name__ == '__main__':
    print('==================================================')
    FaceRecognitionRun().realtime_run()

