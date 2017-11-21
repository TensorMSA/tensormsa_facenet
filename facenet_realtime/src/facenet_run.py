from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import cv2

import numpy as np
import tensorflow as tf
from scipy import misc

import facenet_realtime.src.align.detect_face as detect_face
from facenet_realtime import init_value
from facenet_realtime.src.common import facenet
from facenet_realtime.src.align.align_dataset_rotation import AlignDatasetRotation
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

class DataNodeImage():
    def realtime(self):
        init_value.init_value.init(self)
        # self.realtime_run(self.model_name_detect, 'detect', 'eval')
        # self.realtime_run(self.model_name_rotdet, 'rotdet', 'eval')

        # self.realtime_run(self.model_name_detect, 'detect', 'test')
        self.realtime_run(self.model_name_rotdet, 'rotdet', 'test')

        # self.realtime_run(self.model_name_detect, 'detect', 'real')
        # self.realtime_run(self.model_name_rotdet, 'rotdet', 'real')

    def realtime_run(self, modelName, detectType=None, evalType=None):
        '''
        :param modelName: 사용할 자신의 Classifier Model 
                           self.model_name_detect : detect만 사용한 모델이다.
                           self.model_name_rotdet : Rotate->Detect를 사용한 모델이다.
        :param detectType: 예측할 영상의 이미지 전처리 선택
                           detect : 예측 영상의 얼굴 Detect만 한다.
                           rotdet : 예측 영상의 얼굴 Rotate->Detect를 한다.
        :param evalType: 출력을 어떻게 지정할 지 선택을 해준다.
                         eval : eval_data에 있는 전체 데이터를 가져와 평가를 수행한다.
                         test : eval_data에 있는 폴더별 첫번째 데이터만 예측하여 보여준다. 
                         real : 실제 영상을 예측하여 보여준다.
        :return: 
        '''
        self.detectType = detectType
        self.evalType = evalType

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, self.dets_path)

                # get Model Path
                facenet.get_pre_model_path(self.pre_model_url, self.pre_model_zip, self.model_path, self.pre_model_name)
                facenet.load_model(self.pre_model_name)

                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.embedding_size = self.embeddings.get_shape()[1]
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                classifier_filename = modelName
                classifier_filename_exp = os.path.expanduser(classifier_filename)
                with open(classifier_filename_exp, 'rb') as infile:
                    (self.model, class_names) = pickle.load(infile)
                    print('load classifier file-> %s' % classifier_filename_exp)
                    print('')
                # Sort를 해주는 이유는 기존에 Sequnce를 위해 앞에 붙여둔 것을 제거 하기 위함이다.(L00001_)
                self.HumanNamesSort = sorted(os.listdir(self.train_data_path))
                self.HumanNames = []
                for h in self.HumanNamesSort:
                    h_split = h.split('_')
                    self.HumanNames.append(h_split[1])

                self.predictor, self.detector = AlignDatasetRotation().face_rotation_predictor_download()

                if evalType == "eval":
                    self.facenet_eval(sess)
                else:
                    self.facenet_capture(sess)

    def getpredict(self, sess, frameArr):
        best_class = []
        best_class_box = []
        best_class_boxR = []
        if self.detectType == 'rotdet':
            frameArr, imageFA, best_class_boxR = AlignDatasetRotation.face_rotation(self, frameArr[0], self.predictor, self.detector, 'Y')

            if len(frameArr) == 0:
                best_class.append(-3)
                best_class_box.append([0, 0, 0, 0])
                best_class_boxR.append([0, 0, 0, 0])

        for frame in frameArr:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize frame (optional)

            if frame.ndim == 2:
                frame = facenet.to_rgb(frame)
            frame = frame[:, :, 0:3]

            bounding_boxes, _ = detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet,
                                                        self.threshold, self.factor)
            nrof_faces = bounding_boxes.shape[0]

            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(frame.shape)[0:2]

                cropped = []
                scaled = []
                scaled_reshape = []
                bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                for i in range(nrof_faces):
                    emb_array = np.zeros((1, self.embedding_size))

                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]

                    # inner exception
                    if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                        best_class.append(-1)
                        best_class_box.append([0,0,0,0])
                        continue

                    cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                    cropped[0] = facenet.flip(cropped[0], False)
                    scaled.append(
                        misc.imresize(cropped[0], (self.out_image_size, self.out_image_size), interp='bilinear'))
                    scaled[0] = cv2.resize(scaled[0], (self.image_size, self.image_size),
                                           interpolation=cv2.INTER_CUBIC)

                    scaled[0] = facenet.prewhiten(scaled[0])
                    scaled_reshape.append(scaled[0].reshape(-1, self.image_size, self.image_size, 3))
                    feed_dict = {self.images_placeholder: scaled_reshape[0], self.phase_train_placeholder: False}
                    emb_array[0, :] = sess.run(self.embeddings, feed_dict=feed_dict)
                    predictions = self.model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class.append(best_class_indices[0])
                    best_class_box.append([bb[i][0],bb[i][1],bb[i][2],bb[i][3]])

            else:
                best_class.append(-2)

        i = 0
        if self.detectType == 'detect':
            for bc in best_class:
                result_names = self.HumanNames[bc]
                cv2.rectangle(frame, (best_class_box[i][0], best_class_box[i][1]), (best_class_box[i][2], best_class_box[i][3]), self.box_color, 1)
                if bc >= 0:
                    # cv2.putText(frame, result_names, (best_class_box[i][0], best_class_box[i][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    #         1, self.text_color, thickness=1, lineType=1)
                    frame = Image.fromarray(np.uint8(frame))
                    draw = ImageDraw.Draw(frame)
                    font = ImageFont.truetype(self.font_location, 16)
                    draw.text((best_class_box[i][0], best_class_box[i][1]), result_names, self.text_color, font=font)
                    frame = np.array(frame)

        elif self.detectType == 'rotdet' and len(best_class) == len(best_class_boxR):
            for bc in best_class:
                result_names = self.HumanNames[bc]
                if bc >= 0:
                    # cv2.putText(imageFA, result_names, (best_class_boxR[i][0], best_class_boxR[i][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    #             1, self.text_color, thickness=1, lineType=1)
                    imageFA = Image.fromarray(np.uint8(imageFA))
                    draw = ImageDraw.Draw(imageFA)
                    font = ImageFont.truetype(self.font_location, 16)
                    draw.text((best_class_boxR[i][0], best_class_boxR[i][1]), result_names, self.text_color, font=font)
                    imageFA = np.array(imageFA)

                frame = imageFA

            i += 1

        return best_class, frame

    def facenet_capture(self, sess):
        testCnt = 0
        while True:
            video_capture = cv2.VideoCapture(0)
            ret, frame = video_capture.read()
            if self.evalType == 'test':
                if testCnt < len(self.test_data_files):
                    frame = misc.imread(self.test_data_files[testCnt])
                    testCnt += 1
                else:
                    break

            frameArr = [frame]

            pred, frame = self.getpredict(sess, frameArr)

            if self.evalType == 'real':
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            elif self.evalType == 'test':
                import matplotlib.pyplot as plt
                plt.imshow(frame)
                plt.show()

        if self.evalType == 'real':
            video_capture.release()
            cv2.destroyAllWindows()

    def facenet_eval(self, sess):
        if not os.path.exists(self.eval_data_path):
            print('Eval File is not exists')
            return

        total_class = 0
        total_true = 0
        total_false = 0
        total_none = 0
        result = []

        evaldirlist = sorted(os.listdir(self.eval_data_path))
        for evaldir in evaldirlist:
            evalfile_path = self.eval_data_path+evaldir
            evalfile_list = os.listdir(evalfile_path)
            total_class += 1
            true_cnt = 0
            false_cnt = 0
            none_cnt = 0
            for evalfile in evalfile_list:
                self.evalfile_path = evalfile_path
                self.evalfile = evalfile
                frameArr = [misc.imread(evalfile_path + '/' + evalfile)]
                pred, _ = self.getpredict(sess, frameArr)
                # print(evalfile)
                # print(pred)

                for predcnt in pred:
                    if predcnt == -1 or predcnt == -2 or predcnt == -3:
                        none_cnt += 1
                        total_none += 1
                        self.pred_msg(predcnt, evalfile_path, evalfile)
                    elif evaldir == self.HumanNamesSort[predcnt]:
                        true_cnt += 1
                        total_true += 1
                    else :
                        if self.debug == True:
                            lable = self.HumanNamesSort[predcnt]
                            if len(pred) > 1:
                                print('FalseMulty :'+evalfile_path+'/'+evalfile+' [ True='+evaldir+', Predict='+lable+' ]')
                            else:
                                print('False :'+evalfile_path+'/'+evalfile+' [ True='+evaldir+', Predict='+lable+' ]')

                        false_cnt += 1
                        total_false += 1

            result.append([evaldir, true_cnt+false_cnt, true_cnt, false_cnt, 0, none_cnt])

        if total_true+total_false > 0:
            acc = round((total_true/(total_true+total_false))*100,2)
            resultTotal= ['Total', total_class, total_true, total_false, acc, total_none]

            print('==================================================================')
            print(resultTotal[0]
                  + ' Class :' + str(resultTotal[1])
                  + ', Total Cnt:' + str(resultTotal[2]+resultTotal[3])
                  + ', True Cnt:'+str(resultTotal[2])
                  + ', False Cnt:' + str(resultTotal[3])
                  + ', Accracy:' + str(resultTotal[4]) + '%'
                  + ', None:' + str(resultTotal[5])
                  )

        if true_cnt + false_cnt > 0:
            for i in result:
                acc = round((i[2] / (i[2] + i[3])) * 100, 2)
                print('------------------------------------------------------------------')
                print(i[0]+ ' [ Total Cnt:'+str(i[1])
                          + ', True Cnt:'+str(i[2])
                          + ', False Cnt:'+str(i[3])
                          + ', Accracy:'+str(acc)+' %'
                          + ', None:' + str(i[5])
                          + ' ]')
            print('==================================================================')
            print('')

    def pred_msg(self, predcnt, evalfile_path, evalfile):
        lable = 'face is inner of range!'
        if predcnt == -2:
            lable = 'Unable to align'
        elif predcnt == -3:
            lable = 'Lotation Run Error'

        if self.recogmsg == True:
            print('[ ' + lable + ' ]' + evalfile_path + '/' + evalfile)

if __name__ == '__main__':
    # object detect
    DataNodeImage().realtime()

# [print(f.name) for f in matplotlib.font_manager.fontManager.ttflist if 'Nanum' in f.name]
#
# font_name = fm.FontProperties(fname=self.font_location).get_name()
# matplotlib.rc('font', family=font_name)
# matplotlib.pyplot.imshow(frame)
# matplotlib.pyplot.show()


# self.font_location = self.project_path + 'font/ttf/NanumGothicBold.ttf'
# font_name = font_manager.FontProperties(fname=self.font_location).get_name()
# rc('font', family=font_name)
#
# prop = font_manager.FontProperties(fname=self.font_location, size=18)