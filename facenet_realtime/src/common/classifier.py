from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.svm import SVC

from facenet_realtime import init_value
from facenet_realtime.src.common import facenet
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

class ClassifierImage():
    def classifier_dataset(self, data_path, modelName):
        init_value.init_value.init(self)
        with tf.Graph().as_default():
            with tf.Session() as sess:
                dataset = facenet.get_dataset(data_path)
                paths, labels = facenet.get_image_paths_and_labels(dataset)
                print('Number of classes: %d' % len(dataset))
                print('Number of images: %d' % len(paths))

                print('Loading feature extraction model')
                # get Model Path
                facenet.get_pre_model_path(self.pre_model_url, self.pre_model_zip, self.model_path, self.pre_model_name)
                facenet.load_model(self.pre_model_name)

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                # Run forward pass to calculate embeddings
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                print('Calculating features for images:(' + str(len(range(nrof_batches_per_epoch))) + ')')
                for i in range(nrof_batches_per_epoch):
                    print('features :' + str(i))
                    start_index = i * self.batch_size
                    end_index = min((i + 1) * self.batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, self.image_size)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                classifier_filename = modelName
                classifier_filename_exp = os.path.expanduser(classifier_filename)

                # Train classifier
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)

                # Create a list of class names
                class_names = [cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

    def roc(self, data_path, modelName, eval_path):
        init_value.init_value.init(self)
        with tf.Graph().as_default():
            with tf.Session() as sess:
                dataset = facenet.get_dataset(data_path)
                paths, labels = facenet.get_image_paths_and_labels(dataset)
                print('Number of classes: %d' % len(dataset))
                print('Number of images: %d' % len(paths))

                eval_dataset = facenet.get_dataset(eval_path)
                eval_paths, eval_labels = facenet.get_image_paths_and_labels(eval_dataset)
                print('Number of classes: %d' % len(eval_dataset))
                print('Number of images: %d' % len(eval_paths))

                eval_labels = np.asarray(eval_labels)
                classes = []
                for i in range(len(dataset)):
                    classes.append(i)
                eval_labels = label_binarize(eval_labels, classes=classes)

                print('Loading feature extraction model')
                # get Model Path
                facenet.get_pre_model_path(self.pre_model_url, self.pre_model_zip, self.model_path, self.pre_model_name)
                facenet.load_model(self.pre_model_name)

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                # Run forward pass to calculate embeddings
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                print('Calculating features for images:(' + str(len(range(nrof_batches_per_epoch))) + ')')
                for i in range(nrof_batches_per_epoch):
                    print('features :' + str(i))
                    start_index = i * self.batch_size
                    end_index = min((i + 1) * self.batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, self.image_size)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                # eval_nrof_images = len(eval_paths)
                # eval_nrof_batches_per_epoch = int(math.ceil(1.0 * eval_nrof_images / self.batch_size))
                # eval_emb_array = np.zeros((eval_nrof_images, embedding_size))
                # print('Calculating features for images:(' + str(len(range(eval_nrof_batches_per_epoch))) + ')')
                # for i in range(eval_nrof_batches_per_epoch):
                #     print('features :' + str(i))
                #     start_index = i * self.batch_size
                #     end_index = min((i + 1) * self.batch_size, eval_nrof_images)
                #     paths_batch = eval_paths[start_index:end_index]
                #     images = facenet.load_data(paths_batch, False, False, self.image_size)
                #     feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                #     eval_emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                classifier_filename = modelName
                classifier_filename_exp = os.path.expanduser(classifier_filename)

                # Train classifier
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)

                # Learn to predict each class against the other
                random_state = np.random.RandomState(0)
                classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                                         random_state=random_state))
                #y_score = classifier.fit(emb_array, labels).decision_function(eval_emb_array)
                y_score = classifier.fit(emb_array, labels).decision_function(emb_array)

                # Compute ROC curve and ROC area for each class
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                #for i in range(n_classes):
                fpr[len(dataset)-1], tpr[len(dataset)-1], _ = roc_curve(eval_labels[:, len(dataset)-1], y_score[:, len(dataset)-1])
                roc_auc[len(dataset)-1] = auc(fpr[len(dataset)-1], tpr[len(dataset)-1])

                # Compute micro-average ROC curve and ROC area
                fpr["micro"], tpr["micro"], _ = roc_curve(eval_labels.ravel(), y_score.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                plt.figure()
                lw = 2
                plt.plot(fpr[len(dataset)-1], tpr[len(dataset)-1], color='darkorange',
                         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[len(dataset)-1])
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic example')
                plt.legend(loc="lower right")
                plt.show()






