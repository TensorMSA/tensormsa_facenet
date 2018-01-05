from facenet_tf import init_value
from facenet_tf.src.align.align_dataset_mtcnn import main as align_main
from facenet_tf.src.common.classifier import main as classifier_main
import facenet_tf.src.common.utils as utils

class align_init():
    def __init__(self):
        init_value.init_value.init(self)

class Facenet_run():
    def run(self):
        self = align_init()

        # object detect
        align_main(self)

        # get pre Model Path

        pre_model_url = 'https://drive.google.com/uc?id=0B5MzpY9kBtDVZ2RpVDYwWmxoSUk&export=download'
        pre_model_zip = self.project_dir + 'pre_model/20170512-110547.zip'

        utils.get_pre_model_path(pre_model_url, pre_model_zip, self.model_dir)

        # # classifier Train
        classifier_main(self)

if __name__ == '__main__':
    Facenet_run().run()



