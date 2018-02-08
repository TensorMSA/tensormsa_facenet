import os
import facenet_tf.src.common.download_and_extract as download_and_extract
import facenet_tf.src.common.utils as utils

class init_value():
    def init(self):
        # Common
        self.debug = False  # 이미지 Log를 볼때 사용한다.
        self.gallery_load_flag = True

        self.rotation = False
        self.pair_type = 'svm_pair'  # svm, svm_pair, cnn_pair, distance

        self.project_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        self.gpu_memory_fraction = 0.5
        self.minsize = 20 # minimum size of face
        self.threshold = [0.6, 0.7, 0.7] # three steps's threshold
        self.factor = 0.9 # scale factor 0.709
        self.margin = 0
        self.batch_size = 70
        self.image_size = 160
        self.cropped_size = 25  # rotation use
        self.seed = 666

        # Make Directory
        self.train_dir = utils.make_dir(self.project_dir + 'data/train_data/')
        self.train_detect_dir = utils.make_dir(self.project_dir + 'data/train_detect/')
        self.train_rotation_dir = utils.make_dir(self.project_dir + 'data/train_rotation/')
        self.eval_dir = utils.make_dir(self.project_dir + 'data/eval_data/')

        self.gellery_dir = utils.make_dir(self.project_dir + 'data/gallery_data/')
        self.gallery_detect_dir = utils.make_dir(self.project_dir + 'data/gallery_detect/')
        self.gallery_rotation_dir = utils.make_dir(self.project_dir + 'data/gallery_rotation/')

        self.pre_model_dir = utils.make_dir(self.project_dir + 'pre_model/')

        self.save_dir = utils.make_dir('/hoya_src_root/save_data/')
        self.log_dir = utils.make_dir('/hoya_src_root/log_data/')
        self.lfw_dir = '' # utils.make_dir(self.project_dir + 'data/lfw/')

        # Model
        self.pretrained_model_dir = '20170512-110547'
        self.pretrained_model = self.pre_model_dir + self.pretrained_model_dir + '/' + self.pretrained_model_dir + '.pb'
        if not os.path.exists(self.pretrained_model):
            download_and_extract.download_and_extract_file(self.pretrained_model_dir, self.pre_model_dir)

        self.feature_model_dir = '20170512-110547'
        self.feature_model = self.pre_model_dir + self.feature_model_dir + '/' + self.feature_model_dir + '.pb'

        self.predictor_68_face_landmarks = 'shape_predictor_68_face_landmarks.dat.bz2'
        utils.shape_predictor_68_face_landmarks_download(self.pre_model_dir, self.predictor_68_face_landmarks)

        # file name
        self.classifier_filename = self.pre_model_dir + 'my_classifier.pkl'
        self.gallery_filename = self.pre_model_dir + self.feature_model_dir + '_gallery'
        self.font_location = self.project_dir + 'font/ttf/NanumBarunGothic.ttf'




