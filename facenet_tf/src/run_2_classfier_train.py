from facenet_tf import init_value
from facenet_tf.src.common.classifier import main as classifier_main
import facenet_tf.src.common.utils as utils

class Facenet_run():
    def run(self):
        init_value.init_value.init(self)

        self.use_split_dataset = False
        self.mode = 'TRAIN'
        self.model = self.feature_model
        self.data_dir = self.train_detect_dir
        if self.rotation == True:
            self.data_dir = self.train_rotation_dir

        # classifier Feature Train
        classifier_main(self)

if __name__ == '__main__':
    Facenet_run().run()



