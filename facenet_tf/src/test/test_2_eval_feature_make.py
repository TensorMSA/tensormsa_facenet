from facenet_tf import init_value
from facenet_tf.src.common.classifier import main as classifier_main
import facenet_tf.src.common.utils as utils

class Facenet_run():
    def run(self):
        init_value.init_value.init(self)
        self.model = self.feature_model
        self.output_dir = self.eval_detect_dir
        if self.rotation == True:
            self.output_dir = self.eval_rotation_dir
        self.gallery_filename = self.gallery_eval
        utils.make_feature(self)

if __name__ == '__main__':
    Facenet_run().run()



