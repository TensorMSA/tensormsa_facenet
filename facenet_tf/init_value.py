import os

class init_value():
    def init(self):
        # Common
        self.project_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        self.gpu_memory_fraction = 0.8
        self.minsize = 20 # minimum size of face
        self.threshold = [0.6, 0.7, 0.7] # three steps's threshold
        self.factor = 0.709 # scale factor
        self.margin = 0
        self.image_size = 160

        # Align
        self.input_dir   = self.project_dir + 'data/train_data/'
        self.output_dir  = self.project_dir + 'data/detect_data/'
        self.save_dir    = self.project_dir + 'data/save_data/'
        self.random_order = False  # random.shuffle(dataset)
        self.detect_multiple_faces = False

        # Classifier
        self.model_path = self.project_dir + 'pre_model/'
        self.pre_model_url = 'https://drive.google.com/uc?id=0B5MzpY9kBtDVZ2RpVDYwWmxoSUk&export=download'
        self.pre_model_zip = self.project_dir + 'pre_model/20170512-110547.zip'
        self.model = self.project_dir + 'pre_model/20170512-110547/20170512-110547.pb'
        self.classifier_filename = self.project_dir + 'pre_model/my_classifier_detect.pkl'
        self.use_split_dataset = False
        self.data_dir = self.output_dir
        self.mode = 'TRAIN'
        self.seed = 666
        self.batch_size = 90

        # facenet_run
        self.debug = 'N' # 이미지 Log를 볼때 사용한다.
        self.eval_dir = self.project_dir + 'data/eval_data/'
        self.font_location = self.project_dir + 'font/ttf/NanumBarunGothic.ttf'
        self.box_color = (120, 160, 230)
        self.text_color = (0, 255, 0)
        self.readImageSizeX = 0.7
        self.readImageSizeY = 0.7
        self.viewImageSizeX =2
        self.viewImageSizeY =2







