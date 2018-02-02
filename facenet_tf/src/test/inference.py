import face_recognition
import numpy as np
import tensorflow as tf
import facenet_tf.src.common.facenet as facenet
from facenet_tf import init_value

class inference_run():
    def run(self):
        init_value.init_value.init(self)
        inference_run().main(self)

    def main(self, args):

        with tf.Graph().as_default():
            with tf.Session() as sess:
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

                paths = []
                paths.append("289096.png")
                images = facenet.load_data(paths, False, False, args.image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                unknown_array = sess.run(embeddings, feed_dict=feed_dict)

        npzfile = np.load(args.model_dir+'my_gallery_detect.npz')
        emb_array = npzfile['arr_0']
        labels = npzfile['arr_1']
        class_names = npzfile['arr_2']

        # results is an array of True/False telling if the unknown face matched anyone in the known_faces array
        results = face_recognition.compare_faces(emb_array, unknown_array)

        print(class_names[labels[results.index(True)]])
        print("Is the unknown face a new person that we've never seen before? {}".format(not True in results))

inference_run().run()