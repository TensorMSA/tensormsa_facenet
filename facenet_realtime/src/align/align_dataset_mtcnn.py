from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from scipy import misc

import facenet_realtime.src.align.detect_face as detect_face
from facenet_realtime import init_value
from facenet_realtime.src.common import facenet

class AlignDatasetMtcnn():
    def align_dataset(self, input_path, output_path):
        init_value.init_value.init(self)

        dataset = facenet.get_dataset(input_path)
        output_dir = os.path.expanduser(output_path)

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, self.dets_path)

        # Add a random key to the filename to allow alignment using multiple processes
        random_key = np.random.randint(0, high=99999)
        bounding_boxes_filename = os.path.join(output_dir, self.bounding_boxes+'_%05d.txt' % random_key)

        with open(bounding_boxes_filename, "w") as text_file:
            nrof_images_total = 0
            nrof_successfully_aligned = 0
            for cls in dataset:
                output_class_dir = os.path.join(output_dir, cls.name)
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)
                for image_path in cls.image_paths:
                    nrof_images_total += 1
                    filename = os.path.splitext(os.path.split(image_path)[1])[0]
                    output_filename = os.path.join(output_class_dir, filename + '.png')

                    if not os.path.exists(output_filename):
                        try:
                            img = misc.imread(image_path)
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = '{}: {}'.format(image_path, e)
                            print(errorMessage)
                        else:
                            if img.ndim < 2:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))
                                continue
                            if img.ndim == 2:
                                img = facenet.to_rgb(img)
                                print('to_rgb data dimension: ', img.ndim)
                            img = img[:, :, 0:3]

                            bounding_boxes, _ = detect_face.detect_face(img, self.minsize, pnet, rnet, onet, self.threshold, self.factor)
                            nrof_faces = bounding_boxes.shape[0]

                            if nrof_faces > 0:
                                det = bounding_boxes[:, 0:4]
                                img_size = np.asarray(img.shape)[0:2]
                                if nrof_faces > 1:
                                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                    img_center = img_size / 2
                                    offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                         (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                    index = np.argmax(
                                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                    det = det[index, :]
                                det = np.squeeze(det)
                                bb_temp = np.zeros(4, dtype=np.int32)
                                if det[0] < 0:
                                    det[0] = 0
                                if det[1] < 0:
                                    det[1] = 0
                                if det[2] < 0:
                                    det[2] = 0
                                if det[3] < 0:
                                    det[3] = 0
                                bb_temp[0] = det[0]
                                bb_temp[1] = det[1]
                                bb_temp[2] = det[2]
                                bb_temp[3] = det[3]

                                cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                                scaled_temp = misc.imresize(cropped_temp, (self.image_size, self.image_size), interp='bilinear')

                                nrof_successfully_aligned += 1
                                misc.imsave(output_filename, scaled_temp)
                                text_file.write(
                                    '%s %d %d %d %d\n' % (output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))
                            else:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))

        print('Total number of images: %d' % nrof_images_total)
        print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

