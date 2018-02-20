import os, wget, bz2, dlib
import numpy as np
import tensorflow as tf
import facenet_tf.src.common.facenet as facenet
import matplotlib.pyplot as plt
from keras.utils import np_utils
import random, copy
import math
from scipy import spatial
import scipy.spatial.distance
from numpy import dot
from numpy.linalg import norm
import operator


# common utils
def shape_predictor_68_face_landmarks_download(down_pred_path, bz_pred_file):
    down_pred_url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
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

def get_images_labels_pair(emb_array, labels, dataset):
    p = 0
    nrof_images = len(labels)
    emb_array_pair = np.zeros((nrof_images * nrof_images, emb_array.shape[1]))
    labels_pair = []
    for i in range(nrof_images):
        for j in range(nrof_images):
            if labels[i] == labels[j]:
                labels_pair.append(0)
            else:
                labels_pair.append(1)
                emb = emb_calc(emb_array[i], emb_array[j])
                emb_array_pair[p] = emb
                p += 1

        print(str(i+1))
    return emb_array_pair, labels_pair

def get_images_labels_pair_same(emb_array, labels, dataset):
    total_cnt = 0
    pair_cnt = 0
    for data in dataset:
        total_cnt += len(data)*len(data)

    nrof_images = len(labels)
    random_cnt = round(total_cnt/nrof_images)
    total_cnt += random_cnt*nrof_images
    emb_array_pair = np.zeros((total_cnt, emb_array.shape[1]))

    labels_pair = []
    for i in range(nrof_images):
        for j in range(nrof_images):
            if labels[i] == labels[j]:
                labels_pair.append(0)
                embv = emb_calc(emb_array[i], emb_array[j])
                emb_array_pair[pair_cnt] = embv
                pair_cnt += 1

        diff_cnt = 0
        emb_copy = copy.deepcopy(emb_array)
        labels_copy = copy.deepcopy(labels)
        while diff_cnt < random_cnt:
            if labels[i] in labels_copy:
                r_index = labels_copy.index(labels[i])
            else:
                r_index = random.choice(list(enumerate(labels_copy)))[0]
                labels_pair.append(1)
                embv = emb_calc(emb_array[i], emb_copy[r_index])
                emb_array_pair[pair_cnt] = embv
                pair_cnt += 1
                diff_cnt += 1

            labels_copy[r_index:r_index + 1] = []
            emb_copy = np.delete(emb_copy, r_index, 0)

        print(str(i+1))
    return emb_array_pair, labels_pair

def get_images_labels_pair_same(emb_array, labels, dataset, class_names):
    middle = len(labels) / 2
    labels_pair = []
    emb_array_pair = []
    i = 0
    for un in dataset:
        if un.name.lower().count('unknown') > 0:
            dataset[i:i+1] = []
        i += 1

    for data in dataset:
        index = labels.index(class_names.index(data.name.replace('_',' ')))
        end = index + len(data)
        for i in range(index,end):
            for j in range(index,end):
                labels_pair.append(0)
                embv = emb_calc(emb_array[i], emb_array[j])
                emb_array_pair.append(embv)
            if index < middle:
                for j in range(index + len(data),end + len(data)):
                    labels_pair.append(1)
                    embv = emb_calc(emb_array[i], emb_array[j])
                    emb_array_pair.append(embv)
            else:
                for j in range(0,len(data)):
                    labels_pair.append(1)
                    embv = emb_calc(emb_array[i], emb_array[j])
                    emb_array_pair.append(embv)
            print(i)
    return emb_array_pair, labels_pair

def make_feature(args):
    with tf.Graph().as_default():

        with tf.Session() as sess:

            np.random.seed(seed=args.seed)

            dataset = facenet.get_dataset(args.output_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

            paths, labels = facenet.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            emb_array = np.zeros((nrof_images, embedding_size))


            for i in range(nrof_images):
                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
                # Mrege --->
                print(str(i + 1) + '/' + str(nrof_images))

            class_names = [cls.name.replace('_', ' ') for cls in dataset]

            np.savez(args.gallery_filename, emb_array, labels, class_names, paths)

def train_weight(emb_array, labels, sess):
    # placeholder is used for feeding data.
    x = tf.placeholder("float", shape=[None, 128],name='x')
    y_target = tf.placeholder("float", shape=[None, 2],name='y_target')

    # all the variables are allocated in GPU memory
    W1 = tf.Variable(tf.zeros([128, 2]), name='W1')
    b1 = tf.Variable(tf.zeros([2]), name='b1')
    y = tf.nn.softmax(tf.matmul(x, W1) + b1, name='y')

    # define the Loss function
    cross_entropy = -tf.reduce_sum(y_target * tf.log(y), name='cross_entropy')

    # define optimization algorithm
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_target, 1))
    # correct_prediction is list of boolean which is the result of comparing(model prediction , data)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # tf.cast() : changes true -> 1 / false -> 0
    # tf.reduce_mean() : calculate the mean

    # create summary of parameters
    tf.summary.histogram('weights_1', W1)
    tf.summary.histogram('y', y)
    tf.summary.scalar('cross_entropy', cross_entropy)
    merged = tf.summary.merge_all()

    # Create Session
    # sess = tf.Session(config=tf.ConfigProto(
    #     gpu_options=tf.GPUOptions(allow_growth=True)))  # open a session which is a envrionment of computation graph.
    sess.run(tf.global_variables_initializer())  # initialize the variables

    saver = tf.train.Saver()

    summary_writer = tf.summary.FileWriter("/tmp/mlp", sess.graph)

    labels = np_utils.to_categorical(labels,2)
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    j = 0
    # training the MLP
    for i in range(5001):  # minibatch iteraction
        # batch = mnist.train.next_batch(100)  # minibatch size
        x_batch = emb_array[j:j+100]
        y_batch = labels[j:j+100]
        sess.run(train_step, feed_dict={x: x_batch,
                                        y_target: y_batch}) # placeholder's none length is replaced by i:i+100 indexes

        if i % 500 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: x_batch, y_target: y_batch})
            print("step %d, training accuracy: %.3f" % (i, train_accuracy))

            # calculate the summary and write.
            # summary = sess.run(merged, feed_dict={x: x_batch, y_target: y_batch})
            # summary_writer.add_summary(summary, i)
        j = j + 100
        if (j+100) > len(emb_array)/128:j=0

    # for given x, y_target data set
    # print("test accuracy: %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_target: mnist.test.labels}))
    saver.save(sess, "/home/dev/tensormsa_facenet/facenet_tf/pre_model/my_feature_weight/model.ckpt")
    return sess

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def emb_calc(vector, matrix, type = None):
    emb = vector * matrix

    if type == 'sub':
        emb = 1 - np.sqrt(np.sum(np.square(vector - matrix), axis=1))
    elif type == 'cos':# cosine_similarity
        emb = ( np.sum(vector*matrix,axis=1) / ( np.sqrt(np.sum(matrix**2,axis=1)) * np.sqrt(np.sum(vector**2,axis=1)) ) )[::-1]

    return emb

def get_npz_file(filename):
    emb_array = ''
    emb_labels = ''
    class_names = ''
    file_pathes = ''

    if os.path.exists(filename + '.npz'):
        pairfile = np.load(filename + '.npz')
        emb_array = pairfile['arr_0']
        emb_labels = pairfile['arr_1']
        class_names = pairfile['arr_2']
        file_pathes = pairfile['arr_3']

    return emb_array, emb_labels, class_names, file_pathes

# npzfile = np.load('/home/dev/tensormsa_facenet/facenet_tf/pre_model/20170512-110547_gallery_bak.npz')
# emb_array = npzfile['arr_0']
# labels = npzfile['arr_1'].tolist()
# class_names = npzfile['arr_2'].tolist()
# dataset = facenet.get_dataset('/home/dev/tensormsa_facenet/facenet_tf/data/gallery_detect/')
# end = 0
# batch_size = 1000
# sess = tf.Session(config=tf.ConfigProto(
#         gpu_options=tf.GPUOptions(allow_growth=True)))  # open a session which is a envrionment of computation graph.
#
# for i in range(0,math.ceil(len(dataset) / batch_size)):
#     if i + batch_size > len(dataset):end=len(dataset)
#     else:end=i+batch_size
#     dataset_part = dataset[i:end]
#     emb_array1,labels1 = get_images_labels_pair_same(emb_array,labels,dataset_part,class_names)
#     sess = train_weight(emb_array1,labels1,sess)
# sess.close()