import os, wget, bz2, dlib
import numpy as np
import tensorflow as tf
import facenet_tf.src.common.facenet as facenet

# common utils
def face_rotation_predictor_download(self):
    down_pred_url = self.down_land68_url
    bz_pred_file = self.land68_file
    down_pred_path = self.model_dir
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

def get_images_labels_pair(emb_array, labels):
    p = 0
    nrof_images = len(labels)
    emb_array_pair = np.zeros((nrof_images * nrof_images, emb_array.shape[1]))
    labels_pair = []
    for i in range(nrof_images):
        for j in range(nrof_images):
            embsub = np.subtract(emb_array[i], emb_array[j])
            emb_array_pair[p] = embsub
            p += 1

            if labels[i] == labels[j]:
                labels_pair.append(0)
            else:
                labels_pair.append(1)
        print(str(i+1))
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

            np.savez(args.gallery_filename, emb_array, labels, class_names)


