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
                        frame = cv2.flip( frame, 1 )

                        try:
                            pred, frame = self.getpredict(sess, frame)
                        except:
                            None

                        frame = cv2.resize(frame, (0, 0), fx=self.viewImageSizeX, fy=self.viewImageSizeY)
                        cv2.imshow('Video', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    video_capture.release()
                    cv2.destroyAllWindows()

    def getpredict(self, sess, frame):
        min_box = round(frame.shape[1]/10)
        stand_box = []
        stand_box.append(round(frame.shape[1]/4))
        stand_box.append(round(frame.shape[0]/8))
        stand_box.append(round(frame.shape[1]/4)*3)
        stand_box.append(round(frame.shape[0]/5)*4)
        self.draw_border(frame, (stand_box[0], stand_box[1]), (stand_box[2], stand_box[3]), self.stand_box_color, 2, 10, 20)

        saveframe = frame
        frame = cv2.resize(frame, (0, 0), fx=self.readImageSizeX, fy=self.readImageSizeY)
        img_size = np.asarray(frame.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)

        msgType = 0
        boxes = []
        for i in range(len(bounding_boxes)):
            det = np.squeeze(bounding_boxes[i, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - self.margin / 2, 0)
            bb[1] = np.maximum(det[1] - self.margin / 2, 0)
            bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])

            # if bb[2] - bb[0] > self.boxes_size[0] and bb[2] - bb[0] < self.boxes_size[1]:
            #     boxes.append(bb)
            if min_box > bb[2] - bb[0]:
                msgType = 1 # text = '가까이 다가와 주세요.'
                break

            if stand_box[0] < bb[0] and stand_box[2] > bb[2] and stand_box[1] < bb[1] and stand_box[3] > bb[3]:
                boxes.append(bb)
            else:
                msgType = 2 # text = '박스 안으로 움직여 주세요.'

        if len(boxes) == 0 or len(boxes) > 1:
            if len(boxes) > 1:
                msgType = 3 # text = '한 명만 인식할 수 있습니다.'

            self.reset_list(self.findlist)

        if msgType != 0:
            return _, self.draw_text(frame, self.set_msg(msgType), stand_box)

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
            frame = Image.fromarray(np.uint8(frame))
            draw = ImageDraw.Draw(frame)
            font = ImageFont.truetype(self.font_location, 16)

            # log
            parray = []
            for pcnt in range(len(predictions[0])):
                if predictions[0][pcnt] < 0.05:
                    continue
                parray.append(str(predictions[0][pcnt])[:4]+'_'+self.class_names[pcnt])
            parray.sort(reverse=True)
            print(parray)

            resultFlag = 'Y'
            result = self.class_names[best_class_indices[0]]
            result_names = result + '(' + str(best_class_probabilities[0])[:5] + ')'
            if self.class_names[best_class_indices[0]].lower().find('unknown') > -1:
                for rcnt in range(len(self.findlist)):
                    if self.findlist[rcnt] == '':
                        resultFlag = 'N'
                        break

                if resultFlag == 'Y':
                    result = self.findlist[rcnt]
                    result_names = result + '(' + str(best_class_probabilities[0])[:5] + ')'

            draw.text((boxes[0][0], boxes[0][1]-15), result_names, self.text_color, font=font)
            font = ImageFont.truetype(self.font_location, 20)
            result_names = result + ' 님 인증 되었습니다.'
            draw.text((stand_box[0], stand_box[1] - 30), result_names, self.text_color, font=font)
            frame = np.array(frame)

        cv2.rectangle(frame, (boxes[0][0], boxes[0][1]), (boxes[0][2], boxes[0][3]), self.box_color, 1)

        return _, frame

    def set_msg(self, msgType):
        if msgType == 1:
            text = '가까이 다가와 주세요.'
        elif msgType == 2:
            text = '박스 안으로 움직여 주세요.'
        elif msgType == 3:
            text = '한 명만 인식할 수 있습니다.'

        return text

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

    def draw_text(self, frame, text, boxes):
        frame = Image.fromarray(np.uint8(frame))
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype(self.font_location, 32)
        draw.text((boxes[0], boxes[1] - 30), text, self.alert_color, font=font)
        frame = np.array(frame)
        return frame

    def save_image(self, frame, result_names, result_percent):
        folder = self.save_dir + result_names.replace(' ', '_') + '/'
        filename = result_names + result_percent
        now = datetime.datetime.now()
        nowtime = now.strftime('%Y%m%d%H%M%S')

        if result_names.find('Unknown') == -1:
            if not os.path.exists(folder):
                os.makedirs(folder)
            cv2.imwrite(folder+filename+'.png', frame)
        else:
            filename = str(now) + '_'+ result_percent
            if not os.path.exists(folder):
                os.makedirs(folder)
            cv2.imwrite(folder + filename + '.png', frame)

    # 1 second save
    def save_image_time(self, frame, result_names, result_percent):
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

