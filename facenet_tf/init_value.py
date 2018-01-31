import os
import facenet_tf.src.common.download_and_extract as download_and_extract
import facenet_tf.src.common.utils as utils

class init_value():
    def init(self):
        # Common
        self.project_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        self.gpu_memory_fraction = 0.8
        self.minsize = 20 # minimum size of face
        self.threshold = [0.6, 0.7, 0.7] # three steps's threshold
        self.factor = 0.9 # scale factor 0.709
        self.margin = 0
        self.image_size = 160

        # Align
        self.rotation = False
        self.input_dir   = self.project_dir + 'data/train_data/'
        self.model_dir = self.project_dir + 'pre_model/'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if self.rotation == True:
            self.output_dir = self.project_dir + 'data/train_rotation/'
            self.classifier_filename = self.model_dir + 'my_train_rotation.pkl'
        else:
            self.output_dir  = self.project_dir + 'data/train_detect/'
            self.classifier_filename = self.model_dir + 'my_train_detect.pkl'
        self.save_dir    = '/hoya_src_root/save_data/'
        self.random_order = False  # random.shuffle(dataset)
        self.detect_multiple_faces = False

        # Classifier
        self.pair = True
        self.gallery_filename = self.model_dir + 'my_gallery_detect.pkl'

        # get pre Model Down
        pre_model_name = '20170512-110547'
        self.model = self.model_dir + pre_model_name+'/'+pre_model_name+'.pb'
        download_and_extract.download_and_extract_file(pre_model_name, self.model_dir)

        self.down_land68_url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        self.land68_file = 'shape_predictor_68_face_landmarks.dat.bz2'
        utils.face_rotation_predictor_download(self)

        self.use_split_dataset = False
        self.data_dir = self.output_dir
        self.mode = 'TRAIN'
        self.seed = 666
        self.batch_size = 900

        # facenet_run
        self.debug = 'N' # 이미지 Log를 볼때 사용한다.
        self.eval_dir = self.project_dir + 'data/eval_data/'
        self.font_location = self.project_dir + 'font/ttf/NanumBarunGothic.ttf'
        self.log_dir = '/hoya_src_root/log_data/'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.stand_box_color = (255, 255, 255)
        self.alert_color = (0, 0, 255)
        self.box_color = (120, 160, 230)
        self.text_color = (0, 255, 0)
        self.prediction_cnt = 5  # 로그를 보여주는 개수를 정함
        self.cropped_size = 25  # rotation use

        self.readImageSizeX = 1
        self.readImageSizeY = 1
        self.viewImageSizeX =3
        self.viewImageSizeY =3

        self.findlist = ['', '', '', '', '']  # 배열에 모두 동일한 값이 들어가야 인증이 됨.
        self.boxes_min = 60 # detect min box size
        self.stand_box = [150, 60]# top left(width, height)
        self.prediction_max = 0.1 # 이 수치 이상 정합성을 보여야 인정 됨.

        # feature train
        self.model_def = 'models.inception_resnet_v1'
        self.logs_base_dir=self.model_dir
        self.models_base_dir=self.model_dir
        self.data_dir= self.output_dir    # '~/datasets/casia/casia_maxpy_mtcnnalign_182_160'
        self.max_nrof_epochs=1
        self.epoch_size=1
        self.embedding_size=128
        self.keep_probability=1.0
        self.weight_decay=0.0
        self.center_loss_factor=0.0
        self.center_loss_alfa=0.95
        self.optimizer='ADAGRAD'
        self.learning_rate=0.1
        self.learning_rate_decay_epochs=100
        self.learning_rate_decay_factor=1.0
        self.moving_average_decay=0.9999
        self.seed=666
        self.nrof_preprocess_threads=4
        self.learning_rate_schedule_file='data/learning_rate_schedule.txt'
        self.filter_filename=''
        self.filter_percentile=100.0
        self.filter_min_nrof_images_per_class=0
        self.pretrained_model= self.model

        # Parameters for validation on LFW
        self.lfw_pairs='data/pairs.txt'
        self.lfw_file_ext='png'
        self.lfw_dir=''
        self.lfw_batch_size=100
        self.lfw_nrof_folds=10

        self.random_flip = True
        self.random_rotate = True
        self.random_crop = True
        self.log_histograms = False



