from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from facenet_realtime import init_value
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import wget
import dlib, bz2, cv2
from scipy import misc
import matplotlib.pyplot as plt

class AlignDatasetRotation():
    def rotation_dataset(self, input_path, output_path, resize_flag = None):
        init_value.init_value.init(self)
        images_total = 0
        images_total_rotate = 0
        predictor, detector = self.face_rotation_predictor_download()

        dir_list = os.listdir(input_path)
        for dirList in dir_list:
            if dirList.find(self.bounding_boxes) == 0:
                continue
            file_list = os.listdir(input_path+dirList)
            if not os.path.exists(output_path + dirList):
                os.makedirs(output_path + dirList)

            for img in file_list:
                images_total += 1
                output_filename = output_path + dirList + '/'+ img
                print('rotate('+str(images_total)+'):'+output_filename )
                if not os.path.exists(output_filename):
                    image = [cv2.imread(input_path+'/'+dirList+'/'+img)]

                    try:
                        max_size = max(image[0].shape)

                        if (self.image_resize < max_size and resize_flag == True):
                            image[0] = misc.imresize(image[0], (self.image_resize / max_size))


                        image, _, _ = self.face_rotation(image[0], predictor, detector)
                        for faArr in image:
                            cv2.imwrite(output_filename, faArr)
                        if len(image) > 0:
                            images_total_rotate += 1
                    except:
                        print('Lotation Error:'+input_path+'/'+dirList+'/'+img)

        print('Total number of images: ' + str(images_total))
        print('Number of successfully aligned images: '+str(images_total_rotate))
        print('============================================================================================')

    def face_rotation(self, image, predictor, detector, rectype = None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_boundaries = detector(gray, 2)

        # loop over the face detections
        fa = FaceAligner(predictor, desiredFaceWidth=512)
        faceAligned = []
        best_class_boxR = []
        for rect in face_boundaries:
            (x, y, w, h) = rect_to_bb(rect)

            if rectype == 'Y':
                cv2.rectangle(image, (x, y), (x + w, y + h), self.box_color, 1)

            faceAligned.append(fa.align(image, gray, rect))
            best_class_boxR.append([x, y, x + w, y + h])

        return faceAligned, image, best_class_boxR

    def face_rotation_predictor_download(self):
        init_value.init_value.init(self)
        down_pred_url = self.down_land68_url
        bz_pred_file = self.land68_file
        down_pred_path = self.model_path
        dt_pred_file = bz_pred_file.replace('.bz2', '')

        bz_pred_path = down_pred_path + bz_pred_file
        dt_pred_path = down_pred_path + dt_pred_file

        if not os.path.exists(down_pred_path):
            os.makedirs(down_pred_path)

        if os.path.isfile(bz_pred_path) == False:
            wget.download(down_pred_url, down_pred_path)
            zipfile = bz2.BZ2File(bz_pred_path)  # open the file
            data = zipfile.read()  # get the decompressed data
            newfilepath = bz_pred_path[:-4]  # assuming the filepath ends with .bz2
            open(newfilepath, 'wb').write(data)  # wr

        predictor = dlib.shape_predictor(dt_pred_path)
        detector = dlib.get_frontal_face_detector()
        return predictor, detector



