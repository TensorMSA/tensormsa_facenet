from facenet_tf import init_value
from facenet_tf.src.align.align_dataset_mtcnn import main as align_main
from facenet_tf.src.common.classifier import main as classifier_main
import facenet_tf.src.common.utils as utils

class align_init():
    def __init__(self):
        init_value.init_value.init(self)

class Facenet_run():
    def run(self):
        args = align_init()

        # object detect
        align_main(args)

        # get pre Model Path
        utils.get_pre_model_path(args.pre_model_url, args.pre_model_zip, args.model_path, args.model)

        # # classifier Train
        classifier_main(args)

if __name__ == '__main__':
    Facenet_run().run()



