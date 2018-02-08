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

class FaceRecognition():
    def _init_value(self):
        self.runtype = 'test' # test, real
        self.dettype = 'hog' # dlib, mtcnn, hog, cnn

        self.readImageSizeX = 1
        self.readImageSizeY = 1
        self.viewImageSizeX = 3
        self.viewImageSizeY = 3

        self.findlist = ['', '', '', '', '']  # 배열에 모두 동일한 값이 들어가야 인증이 됨.
        self.boxes_min = 70  # detect min box size
        self.stand_box = [50, 30]  # top left(width, height)
        self.prediction_max = 0.1  # 이 수치 이상 정합성을 보여야 인정 됨.
        self.prediction_cnt = 6  # 로그를 보여주는 개수를 정함

        self.stand_box_color = (255, 255, 255)
        self.alert_color = (0, 0, 255)
        self.box_color = (120, 160, 230)
        self.text_color = (0, 255, 0)

    def realtime_run(self):
        init_value.init_value.init(self)
        self._init_value()

        # GPU
        # with tf.Graph().as_default():
        #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
        #     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        #     with sess.as_default():

        # CPU
        config = tf.ConfigProto( device_count={'GPU': 0} )
        with tf.Session(config=config) as sess:

            self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)

            # Pre Train Load the model
            print('Loading feature extraction model')
            facenet.load_model(self.feature_model)

            # Get input and output tensors
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Model Load
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
            self.fa = FaceAligner(self.predictor, desiredFaceWidth=self.image_size+self.cropped_size)

            if self.runtype == 'test':
                self.total_cnt = 0
                self.total_t_cnt = 0
                self.total_f_cnt = 0
                self.total_u_cnt = 0

                evaldirlist = sorted(os.listdir(self.eval_dir))
                for evalpath in evaldirlist:
                    self.evalpath = evalpath
                    evalfile_path = self.eval_dir + self.evalpath
                    evalfile_list = os.listdir(evalfile_path)
                    test_data_files = []
                    for evalfile in evalfile_list:
                        test_data_files.append(evalfile_path + '/' + evalfile)
                        # break
                    print(evalfile_path)
                    self.file_cnt = 0
                    self.file_t_cnt = 0
                    self.file_f_cnt = 0
                    self.file_u_cnt = 0

                    for test_file in test_data_files:
                        self.testfile = test_file
                        frame = misc.imread(test_file)
                        self.get_pair_file(self.gallery_filename + '.npz')
                        try:
                            frame, self = predict.getpredict(self, sess, frame)
                            # plt.imshow(frame)
                            # plt.show()
                        except:
                            None

                    print('Total:'+str(self.file_cnt)+ ', T:'+str(self.file_t_cnt)+', F:'+str(self.file_f_cnt)+', U:'+str(self.file_u_cnt))
                print('-------------------------------------------------------------')
                print('Total:'+str(self.total_cnt)+ ', T:'+str(self.total_t_cnt)+', F:'+str(self.total_f_cnt)+', U:'+str(self.total_u_cnt))
                print('Total Percent : '+str(round(100-self.total_f_cnt/self.total_cnt*100,4)))
            else:
                self.pretime = '99' # 1 second save
                video_capture = cv2.VideoCapture(0)

                while True:
                    ret, frame = video_capture.read()
                    frame = cv2.flip( frame, 1 )
                    self.get_pair_file(self.gallery_filename + '.npz')

                    frame = predict.getpredict(self, sess, frame)

                    frame = cv2.resize(frame, (0, 0), fx=self.viewImageSizeX, fy=self.viewImageSizeY)
                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                video_capture.release()
                cv2.destroyAllWindows()

    def get_pair_file(self, filename):
        if self.gallery_load_flag:
            pairfile = np.load(filename)
            self.emb_array = pairfile['arr_0']
            self.emb_labels = pairfile['arr_1']
            self.class_names = pairfile['arr_2']
            self.pair_load_flag = False

if __name__ == '__main__':
    print('==================================================')
    FaceRecognition().realtime_run()

