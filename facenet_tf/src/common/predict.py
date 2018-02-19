from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
import facenet_tf.src.align.detect_face as detect_face
from facenet_tf.src.common import facenet
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import datetime
import dlib
from imutils.face_utils import rect_to_bb
import face_recognition
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import facenet_tf.src.common.utils as utils

def getpredict(self, sess, frame):
    saveframe = frame
    frame = cv2.resize(frame, (0, 0), fx=self.readImageSizeX, fy=self.readImageSizeY)

    if self.gallery_load_flag:
        pairfile = np.load(self.gallery_filename + '.npz')
        self.emb_array = pairfile['arr_0']
        self.emb_labels = pairfile['arr_1']
        self.class_names = pairfile['arr_2']

        self.stand_box = [self.stand_box[0], self.stand_box[1]]
        self.stand_box.append(frame.shape[1] - self.stand_box[0])
        self.stand_box.append(frame.shape[0] - self.stand_box[1])

        self.gallery_load_flag = False

    frame = self.draw_border(frame, (self.stand_box[0], self.stand_box[1]), (self.stand_box[2], self.stand_box[3]),
                             self.stand_box_color, 2, 10, 20)

    aligned, boxes = get_boxes_frame(self, frame)

    if len(boxes) == 0:
        reset_list(self.findlist)
        return frame

    # if len(bounding_boxes) > 1:
    #     msgType = 3  # text = '한 명만 인식할 수 있습니다.'

    msgType = 0
    if self.boxes_min > boxes[0][2] - boxes[0][0]:
        msgType = 1  # text = '가까이 다가와 주세요.'
    elif self.stand_box[0] > boxes[0][0] or self.stand_box[2] < boxes[0][2]:
        msgType = 2  # text = '박스 안으로 움직여 주세요.'
    elif self.stand_box[1] > boxes[0][1] or self.stand_box[3] < boxes[0][3]:
        msgType = 2  # text = '박스 안으로 움직여 주세요.'

    if msgType != 0 and self.runtype == 'real':
        frame = draw_text(self, frame, set_predict_msg(msgType), self.stand_box)
        # self.save_image(frame)
        return frame

    cv2.rectangle(frame, (boxes[0][0], boxes[0][1]), (boxes[0][2], boxes[0][3]), self.box_color, 1)

    if self.predict_flag:
        self.predict_flag = False
        return frame
    else:
        self.predict_flag = True

    prewhitened = facenet.prewhiten(aligned)
    prewhitened_reshape = prewhitened.reshape(-1, self.image_size, self.image_size, 3)

    # Run forward pass to calculate embeddings
    feed_dict = {self.images_placeholder: prewhitened_reshape, self.phase_train_placeholder: False}
    self.emb = sess.run(self.embeddings, feed_dict=feed_dict)

    best_class_indices, best_class_probabilities = get_predict_index(self)

    fcnt = 0
    for flist in self.findlist:
        pre = flist
        cur = self.class_names[best_class_indices[0]]
        preun = pre.lower().find('unknown')
        curun = cur.lower().find('unknown')
        if best_class_probabilities[0] > self.prediction_max:
            if curun > -1:
                cur = 'unknown'

            if pre == '' or (preun > -1 and curun == -1):
                self.findlist[fcnt] = cur
                break
            elif pre != cur and curun == -1:
                self.logger.error('====================================================')
                self.logger.error(self.findlist)
                self.logger.error('Current Fail Predict : ' + self.class_names[best_class_indices[0]] + '(' + str(
                    best_class_probabilities[0])[:5] + ')')
                reset_list(self.findlist)
        fcnt += 1
    # print(self.findlist)
    if '' not in self.findlist or self.findlist.count('unknown') == len(self.findlist):
        # save
        save_image(self, saveframe, self.class_names[best_class_indices[0]], str(best_class_probabilities[0])[:5])

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

        frame = Image.fromarray(np.uint8(frame))
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype(self.font_location, self.name_font_size)
        draw.text((boxes[0][0], boxes[0][1] - 15), result_names, self.text_color, font=font)
        font = ImageFont.truetype(self.font_location, self.result_font_size)
        if self.findlist.count('unknown') > 0:
            result_names = ''
        else:
            result_names = result + ' 님 인증 되었습니다.'
            print(result_names)
        draw.text((self.stand_box[0], self.stand_box[1] - self.result_font_size), result_names, self.text_color, font=font)
        frame = np.array(frame)
        reset_list(self.findlist)
        # self.save_image(frame, self.class_names[best_class_indices[0]], str(best_class_probabilities[0])[:5])
    return frame

def getpredict_test(self, sess):
    if os.path.exists(self.gallery_filename + '.npz'):
        pairfile = np.load(self.gallery_filename + '.npz')
        self.emb_array = pairfile['arr_0']
        self.emb_labels = pairfile['arr_1']
        self.class_names = pairfile['arr_2']
        self.file_pathes = pairfile['arr_3']

    if os.path.exists(self.gallery_eval + '.npz'):
        pairfile_eval = np.load(self.gallery_eval + '.npz')
        emb_array_eval = pairfile_eval['arr_0']
        emb_labels_eval = pairfile_eval['arr_1']
        class_names_eval = pairfile_eval['arr_2']
        file_pathes_eval = pairfile_eval['arr_3']

    emb_cnt = 0
    eval_class = []
    eval_true = []
    eval_false = []
    eval_unknown = []

    for emb in emb_array_eval:
        if emb_cnt%self.eval_log_cnt == 0 and emb_cnt != 0:
            set_eval_log(eval_true, eval_false, eval_unknown)

        self.emb = emb
        emb_class = class_names_eval[emb_labels_eval[emb_cnt]]

        if eval_class.count(emb_class) == 0:
            print('---------------------------------------------------------------------------------------------------')
            print('Class Name:'+emb_class)
            print('---------------------------------------------------------------------------------------------------')
            eval_class.append(emb_class)
            eval_true.append(0)
            eval_false.append(0)
            eval_unknown.append(0)

        class_idx = eval_class.index(emb_class)

        best_class_indices, best_class_probabilities = get_predict_index(self)

        if emb_class == self.class_names[best_class_indices]:
            eval_true[class_idx] += 1
        elif self.class_names[best_class_indices][0].lower().count('unknown') > 0:
            eval_unknown[class_idx] += 1
        else:
            eval_false[class_idx] += 1
            print('False:'+file_pathes_eval[emb_cnt])

        emb_cnt += 1

    set_eval_log(eval_true, eval_false, eval_unknown)
    print('---------------------------------------------------------')
    for cnt in range(len(eval_class)):
        total_cnt = eval_true[cnt]+eval_false[cnt]+eval_unknown[cnt]
        if total_cnt == 0:
            continue
        print(eval_class[cnt]+' Total:'+str(total_cnt)
                +' T:'+str(eval_true[cnt])+' F:'+str(eval_false[cnt])+' U:'+str(eval_unknown[cnt])
                + '     Fail:'+str(eval_false[cnt]/total_cnt*100)[:6])

def set_eval_log(eval_true, eval_false, eval_unknown):
    t = sum(eval_true)
    f = sum(eval_false)
    u = sum(eval_unknown)
    total = t+f+u
    if total == 0:
        return

    print('Eval Total:'+str(total)+' T:'+str(t)+' F:'+str(f)+' U:'+str(u)
          + '     Fail:' + str(f / total * 100)[:6])

def get_boxes_frame(self, frame):
    boxes = []
    img_size = np.asarray(frame.shape)[0:2]
    if len(img_size) == 0:
        return frame, boxes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if self.detect_type == 'dlib':
        bounding_boxes = self.detector(gray, 2)
    elif self.detect_type == 'hog' or self.detect_type == 'cnn':
        bounding_boxes = face_recognition.face_locations(frame, number_of_times_to_upsample=0, model=self.detect_type)
    elif self.detect_type == 'mtcnn':
        bounding_boxes, _ = detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet,
                                                    self.threshold, self.factor)

    for bounding_box in bounding_boxes:
        if self.detect_type == 'dlib':
            det = rect_to_bb(bounding_box)
        else:
            det = np.squeeze(bounding_box[0:4])

        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - self.margin / 2, 0)
        bb[1] = np.maximum(det[1] - self.margin / 2, 0)
        bb[2] = np.minimum(det[2] + self.margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + self.margin / 2, img_size[0])

        if self.detect_type == 'dlib':
            bb[2] += bb[0]
            bb[3] += bb[1]
        elif self.detect_type == 'hog' or self.detect_type == 'cnn':
            bb[1], bb[2], bb[3], bb[0] = bounding_box

        if len(boxes) == 0:
            boxes.append(bb)
        else:
            if boxes[0][2] - boxes[0][0] < bb[2] - bb[0]:
                boxes[0] = bb

    if len(boxes) > 0:
        if self.rotation == True:
            rect = dlib.rectangle(left=int(boxes[0][0]), top=int(boxes[0][1]), right=int(boxes[0][2]),
                                  bottom=int(boxes[0][3]))
            cropped = self.fa.align(frame, gray, rect)[self.cropped_size:self.image_size, self.cropped_size:self.image_size,:]
        else:
            cropped = frame[boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2], :]

        frame = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')

    return frame, boxes

def get_predict_index(self):
    parray = []
    log_cnt = 0

    if self.pair_type == 'svm':
        predictions = self.model.predict_proba(self.emb)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        for pcnt in predictions[0].argsort():
            if self.prediction_cnt > log_cnt and predictions[0][pcnt] < 1 and predictions[0][
                pcnt] > self.prediction_max:
                parray.append(str(predictions[0][pcnt])[:7] + '_' + self.class_names[pcnt])
                log_cnt += 1
    elif self.pair_type == 'svm_pair':
        embv = utils.emb_calc(self.emb_array, self.emb)
        predictions = self.model.predict_proba(embv)
        best_class_indices = [self.emb_labels[np.argmax(predictions, axis=0)[0]]]
        best_class_probabilities = np.amax(predictions, axis=0)

        for pcnt in predictions[:, 0].argsort()[::-1]:
            if self.prediction_cnt > log_cnt and predictions[pcnt][0] < 1 and predictions[pcnt][
                0] > self.prediction_max:
                parray.append(str(predictions[pcnt][0])[:7] + '_' + self.class_names[self.emb_labels[pcnt]])
                log_cnt += 1
    elif self.pair_type.count('distance') > 0:
        if self.pair_type.count('cos') > 0:
            predictions = utils.emb_calc(self.emb_array, self.emb, 'cos')
        elif self.pair_type.count('sub') > 0:
            predictions = utils.emb_calc(self.emb_array, self.emb, 'sub')


        distmax = np.argmax(predictions)
        best_class_indices = [self.emb_labels[distmax]]
        best_class_probabilities = [predictions[distmax]]

        for pcnt in predictions.argsort()[::-1]:
            if self.prediction_cnt > log_cnt and predictions[pcnt] > self.prediction_max:
                parray.append(str(predictions[pcnt])[:7] + '_' + self.class_names[self.emb_labels[pcnt]])
                log_cnt += 1

    elif self.pair_type == 'cnn_pair':
        emb_sub = utils.emb_calc(self.emb_array, self.emb)
        with tf.Graph().as_default():
            with tf.Session() as sess:
                # placeholder is used for feeding data.
                x = tf.placeholder("float", shape=[None, 128], name='x')

                # all the variables are allocated in GPU memory
                W1 = tf.Variable(tf.zeros([128, 2]), name='W1')
                b1 = tf.Variable(tf.zeros([2]), name='b1')
                y = tf.nn.softmax(tf.matmul(x, W1) + b1, name='y')
                saver = tf.train.Saver()
                saver.restore(sess, "/home/dev/tensormsa_facenet/facenet_tf/pre_model/my_feature_weight/model.ckpt")
                predictions = sess.run(y, feed_dict={x: emb_sub})
        best_class_indices = [self.emb_labels[np.argmax(predictions, axis=0)[0]]]
        best_class_probabilities = np.amax(predictions, axis=0)

        for pcnt in predictions[:, 0].argsort()[::-1][:self.prediction_cnt]:
            parray.append(str(predictions[pcnt][0])[:7] + '_' + self.class_names[self.emb_labels[pcnt]])

    # print('----------------------------------------------------------------------------------')
    if len(parray) > 1 and self.debug:
        print(parray)
    # print('----------------------------------------------------------------------------------')
    return best_class_indices, best_class_probabilities

def set_predict_msg(msgType):
    if msgType == 1:
        text = '가까이 다가와 주세요.'
    elif msgType == 2:
        text = '박스 안으로 움직여 주세요.'
    elif msgType == 3:
        text = '한 명만 인식할 수 있습니다.'

    return text

def draw_text(self, frame, text, boxes):
    frame = Image.fromarray(np.uint8(frame))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.truetype(self.font_location, self.alert_font_size)
    draw.text((boxes[0], boxes[1] - self.alert_font_size), text, self.alert_color, font=font)
    frame = np.array(frame)
    return frame

def save_image(self, frame, result_names=None, result_percent=None):
    if result_names == None:
        result_names = 'resultNone'
        result_percent = '0'
    folder = self.save_dir + result_names.replace(' ', '_') + '/'
    filename = result_names + result_percent
    now = datetime.datetime.now()
    nowtime = now.strftime('%Y%m%d%H%M%S')

    if result_names.find('Unknown') == -1 and result_names != 'resultNone':
        if not os.path.exists(folder):
            os.makedirs(folder)
        cv2.imwrite(folder+filename+'.png', frame)
    else:
        filename = str(now) + '_'+ result_percent
        if not os.path.exists(folder):
            os.makedirs(folder)
        cv2.imwrite(folder + filename + '.png', frame)

def reset_list(list):
    for i in range(len(list)):
        list[i] = ''

