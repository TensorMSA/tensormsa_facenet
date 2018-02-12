import os
import facenet_tf.src.common.download_and_extract as download_and_extract
import facenet_tf.src.common.utils as utils

class init_value():
    def init(self):
        # Common
        self.debug = True  # 이미지 Log를 볼때 사용한다.
        self.gallery_load_flag = True

        self.rotation = False
        self.detect_type = 'hog' # dlib, mtcnn, hog, cnn
        self.pair_type = 'distance'  # svm, svm_pair, cnn_pair, distance

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
        train_cnt = ''
        eval_cnt = ''
        gallery_cnt = ''
        data_dir = 'data'
        pre_model_dir = 'pre_model'

        self.train_dir = utils.make_dir(self.project_dir + data_dir +'/train_data'+train_cnt+'/')
        self.train_detect_dir = utils.make_dir(self.project_dir + data_dir +'/train_detect'+train_cnt+'/')
        self.train_rotation_dir = utils.make_dir(self.project_dir + data_dir +'/train_rotation'+train_cnt+'/')
        self.eval_dir = utils.make_dir(self.project_dir + data_dir +'/eval_data'+eval_cnt+'/')
        self.eval_detect_dir = utils.make_dir(self.project_dir + data_dir +'/eval_detect'+eval_cnt+'/')
        self.eval_rotation_dir = utils.make_dir(self.project_dir + data_dir +'/eval_rotation'+eval_cnt+'/')

        self.gellery_dir = utils.make_dir(self.project_dir + 'data/gallery_data'+gallery_cnt+'/')
        self.gallery_detect_dir = utils.make_dir(self.project_dir + 'data/gallery_detect'+gallery_cnt+'/')
        self.gallery_rotation_dir = utils.make_dir(self.project_dir + 'data/gallery_rotation'+gallery_cnt+'/')

        self.pre_model_dir = utils.make_dir(self.project_dir + pre_model_dir + '/')

        self.save_dir = utils.make_dir('/hoya_src_root/save_data/')
        self.log_dir = utils.make_dir('/hoya_src_root/log_data/')
        self.lfw_dir = ''
        # self.lfw_dir = utils.make_dir(self.project_dir + 'data/lfw/lfw_mtcnnalign_160/')

        # Model
        self.pretrained_model_dir = '20170512-110547'
        self.pretrained_model = self.pre_model_dir + self.pretrained_model_dir + '/' + self.pretrained_model_dir + '.pb'
        if not os.path.exists(self.pretrained_model):
            download_and_extract.download_and_extract_file(self.pretrained_model_dir, self.pre_model_dir)

        self.feature_model_dir = '20170512-110547' #'20180207-000626'
        self.feature_model = self.pre_model_dir + self.feature_model_dir + '/' + self.feature_model_dir + '.pb'

        self.predictor_68_face_landmarks = 'shape_predictor_68_face_landmarks.dat.bz2'
        utils.shape_predictor_68_face_landmarks_download(self.pre_model_dir, self.predictor_68_face_landmarks)

        # file name
        self.classifier_filename = self.pre_model_dir + self.feature_model_dir+'_my_classifier.pkl'+train_cnt
        self.gallery_filename = self.pre_model_dir + self.feature_model_dir + '_gallery'+gallery_cnt
        self.gallery_eval = self.pre_model_dir + self.feature_model_dir + '_gallery'+'_eval'+eval_cnt
        self.font_location = self.project_dir + 'font/ttf/NanumBarunGothic.ttf'




