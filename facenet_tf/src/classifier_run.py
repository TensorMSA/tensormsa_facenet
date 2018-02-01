from facenet_tf import init_value
from facenet_tf.src.align.align_dataset_mtcnn import main as align_main
from facenet_tf.src.common.classifier import main as classifier_main

class Facenet_run():
    def run(self):
        init_value.init_value.init(self)

        # object detect
        # align_main(self)

        # classifier Train
        classifier_main(self)

if __name__ == '__main__':
    Facenet_run().run()



