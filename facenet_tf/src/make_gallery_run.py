from facenet_tf import init_value
from facenet_tf.src.align.align_dataset_mtcnn import main as align_main
import facenet_tf.src.common.utils as utils
import os

class Facenet_run():
    def run(self):
        init_value.init_value.init(self)

        # object detect
        self.input_dir = self.project_dir + 'data/gallery_data/'
        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)
        self.output_dir = self.project_dir + 'data/gallery_detect/'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        align_main(self)

        # classifier Train
        utils.make_feature(self)

if __name__ == '__main__':
    Facenet_run().run()



