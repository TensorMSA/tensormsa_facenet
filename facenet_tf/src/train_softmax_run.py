from facenet_tf import init_value

from facenet_tf.src.common.train_softmax import main as train_softmax_main

class Facenet_run():
    def run(self):
        init_value.init_value.init(self)
        train_softmax_main(self)

if __name__ == '__main__':
    Facenet_run().run()