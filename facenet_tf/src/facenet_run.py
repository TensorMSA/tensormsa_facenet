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


class DataNodeImage():
    def realtime_run(self, runtype = 'real'):
        init_value.init_value.init(self)

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)

                # Pre Train Load the model
                print('Loading feature extraction model')
                facenet.load_model(self.model)

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

                if runtype == 'test':
                    test_data_files = []
                    evaldirlist = sorted(os.listdir(self.eval_dir))
                    for evalpath in evaldirlist:
                        evalfile_path = self.eval_dir + '/' + evalpath
                        evalfile_list = os.listdir(evalfile_path)

                        for evalfile in evalfile_list:
                            test_data_files.append(evalfile_path + '/' + evalfile)
                            break
                    frame = misc.imread(test_data_files[0])
                    pred, frame = self.getpredict(sess, frame)
                    plt.imshow(frame)
                    plt.show()
                else:
                    self.pretime = '99' # 1 second save
                    video_capture = cv2.VideoCapture(0)

                    while True:
                        ret, frame = video_capture.read()
                        pred, frame = self.getpredict(sess, frame)

                        frame = cv2.resize(frame, (0, 0), fx=self.viewImageSizeX, fy=self.viewImageSizeY)
                        cv2.imshow('Video', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    video_capture.release()
                    cv2.destroyAllWindows()

    def getpredict(self, sess, frame):
        saveframe = frame
        frame = cv2.resize(frame, (0, 0), fx=self.readImageSizeX, fy=self.readImageSizeY)
        img_size = np.asarray(frame.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)

        boxes = []
        for i in range(len(bounding_boxes)):
            det = np.squeeze(bounding_boxes[i, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - self.margin / 2, 0)
            bb[1] = np.maximum(det[1] - self.margin / 2, 0)
            bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])

            if bb[2] - bb[0] > self.boxes_size[0] and bb[2] - bb[0] < self.boxes_size[1]:
                boxes.append(bb)

        if len(boxes) == 0 or len(boxes) > 1:
            if len(boxes) > 1:
                print('Reset : More than one person can not recognize.')
            self.reset_list(self.findlist)
            return _, frame

        cropped = frame[boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2], :]
        aligned = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        prewhitened_reshape = prewhitened.reshape(-1, self.image_size, self.image_size, 3)

        # Run forward pass to calculate embeddings
        feed_dict = {self.images_placeholder: prewhitened_reshape, self.phase_train_placeholder: False}
        emb = sess.run(self.embeddings, feed_dict=feed_dict)

        predictions = self.model.predict_proba(emb)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        try:
            viewFlag = 'Y'
            for fcnt in range(len(self.findlist)):
                if self.findlist[fcnt] == '':
                    self.findlist[fcnt] = self.class_names[best_class_indices[0]]
                    viewFlag = 'N'
                    break

                if self.findlist[fcnt] != self.class_names[best_class_indices[0]]:
                    if self.class_names[best_class_indices[0]].lower().find('unknown') == -1:
                        if self.findlist[fcnt].lower().find('unknown') == -1:
                            self.logger.error(self.findlist)
                            self.logger.error('Failed : '+self.class_names[best_class_indices[0]]+'('+str(best_class_probabilities[0])[:5]+')')
                        self.reset_list(self.findlist)
                        viewFlag = 'N'
        except Exception as e:
            viewFlag = 'N'
            print(e)
        # print(self.findlist)


        if viewFlag == 'Y':
            self.save_image(saveframe, self.class_names[best_class_indices[0]], str(best_class_probabilities[0])[:5])
            # log
            parray = []
            for pcnt in range(len(predictions[0])):
                if predictions[0][pcnt] < 0.05:
                    continue
                parray.append(str(predictions[0][pcnt])[:4]+'_'+self.class_names[pcnt])
            parray.sort(reverse=True)
            print(parray)

            resultFlag = 'Y'
            result_names = self.class_names[best_class_indices[0]] + '(' + str(best_class_probabilities[0])[:5] + ')'
            if self.class_names[best_class_indices[0]].lower().find('unknown') > -1:
                for rcnt in range(len(self.findlist)):
                    if self.findlist[rcnt] == '':
                        resultFlag = 'N'
                        break

                if resultFlag == 'Y':
                    result_names = self.findlist[rcnt] + '(' + str(best_class_probabilities[0])[:5] + ')'

            frame = Image.fromarray(np.uint8(frame))
            draw = ImageDraw.Draw(frame)
            font = ImageFont.truetype(self.font_location, 16)


            draw.text((boxes[0][0], boxes[0][1]-15), result_names, self.text_color, font=font)
            frame = np.array(frame)

        cv2.rectangle(frame, (boxes[0][0], boxes[0][1]), (boxes[0][2], boxes[0][3]), self.box_color, 1)

        return _, frame

    def save_image(self, frame, result_names, result_percent):
        now = datetime.datetime.now()
        nowtime = now.strftime('%Y%m%d%H%M%S')

        if nowtime != self.pretime:
            folder = self.save_dir + result_names.replace(' ', '_') + '/'
            filename = result_names + result_percent
            if result_names.find('Unknown') == -1:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                cv2.imwrite(folder+filename+'.png', frame)
                self.pretime = nowtime
            else:
                if int(nowtime) - int(self.pretime) >= 1:
                    filename = str(nowtime)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    cv2.imwrite(folder + filename + '.png', frame)
                    self.pretime = nowtime

    def reset_list(self, list):
        for i in range(len(list)):
            list[i] = ''

if __name__ == '__main__':
    DataNodeImage().realtime_run('real')

