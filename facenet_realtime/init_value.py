import os

class init_value():
    def init(self):
        # 틀린 이미지를 찾을때 사용한다.
        self.debug = True
        # 인식X or Multy 이미지를 찾을때 사용한다.
        self.recogmsg = False

        self.image_size = 160
        self.batch_size = 1000
        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor
        self.frame_interval = 3
        self.out_image_size = 182

        self.text_color = (255, 0, 0)
        self.box_color =(120, 160, 230)

        self.project_path = os.path.dirname(os.path.abspath(__file__))+'/'
        self.train_data_path = self.project_path+'data/train_data/'
        self.eval_data_path = self.project_path+'data/eval_data/'

        self.detect_data_path = self.project_path+'data/data_detect/'
        self.rotate_data_path = self.project_path+'data/data_rotate/'
        self.rotdet_data_path = self.project_path+'data/data_rotdet/'

        if not os.path.exists(self.detect_data_path):
            os.makedirs(self.detect_data_path)
        if not os.path.exists(self.rotate_data_path):
            os.makedirs(self.rotate_data_path)
        if not os.path.exists(self.rotdet_data_path):
            os.makedirs(self.rotdet_data_path)

        self.model_path = self.project_path + 'pre_model/'
        self.model_name_detect = self.project_path + 'pre_model/my_classifier_detect.pkl'
        self.model_name_rotate = self.project_path + 'pre_model/my_classifier_rotate.pkl'
        self.model_name_rotdet = self.project_path + 'pre_model/my_classifier_rotdet.pkl'

        self.dets_path = self.project_path + 'dets/'

        # facenet object detect
        self.pre_model_url = 'https://drive.google.com/uc?id=0B5MzpY9kBtDVZ2RpVDYwWmxoSUk&export=download'
        self.pre_model_zip = self.project_path + 'pre_model/20170512-110547.zip'
        self.pre_model_name = self.project_path + 'pre_model/20170512-110547/20170512-110547.pb'

        # drip object rotate
        self.down_land68_url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
        self.land68_file = 'shape_predictor_68_face_landmarks.dat.bz2'

        self.test_data_files = []
        evaldirlist = sorted(os.listdir(self.eval_data_path))
        for evaldir in evaldirlist:
            evalfile_path = self.eval_data_path + '/' + evaldir
            evalfile_list = os.listdir(evalfile_path)

            for evalfile in evalfile_list:
                self.test_data_files.append(evalfile_path + '/' + evalfile)
                break

        self.font_location = self.project_path+'font/ttf/NanumBarunGothic.ttf'

        self.bounding_boxes = 'bounding_boxes'
        self.pnet = None
        self.rnet = None
        self.onet = None
        self.images_placeholder = None
        self.embeddings = None
        self.embedding_size = None
        self.phase_train_placeholder = None
        self.evalfile_path = None
        self.evalfile = None



