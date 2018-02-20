from facenet_tf import init_value
from facenet_tf.src.common.validate_on_lfw import main as validate_on_lfw_main
import facenet_tf.src.common.utils as utils

class Facenet_run():
    def run(self):
        init_value.init_value.init(self)
        self.lfw_dir = self.lfw_detect_dir
        if self.rotation == True:
            self.lfw_dir = self.lfw_rotation_dir
        self.model = self.feature_model
        # lfw validation
        validate_on_lfw_main(self)

if __name__ == '__main__':
    Facenet_run().run()