from facenet_tf import init_value
from facenet_tf.src.common.classifier import main as classifier_main

class Facenet_run():
    def run(self):
        init_value.init_value.init(self)

        # classifier Train
        classifier_main(self)

if __name__ == '__main__':
    Facenet_run().run()



