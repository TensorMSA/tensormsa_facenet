from facenet_tf import init_value
from facenet_tf.src.common.train_softmax import main as train_softmax_main
from facenet_tf.src.common.freeze_graph import main as freeze_graph_main

class Facenet_run():
    def run(self):
        init_value.init_value.init(self)

        self.logs_base_dir = self.pre_model_dir
        self.models_base_dir = self.pre_model_dir
        self.data_dir = self.train_detect_dir
        if self.rotation == True:
            self.data_dir = self.train_rotation_dir

        self.max_nrof_epochs = 500
        self.epoch_size = 1000

        self.learning_rate = 0.001
        self.learning_rate_decay_epochs = 100
        self.learning_rate_decay_factor = 1.0
        self.moving_average_decay = 0.9999
        self.optimizer = 'ADAM'
        self.keep_probability = 0.8
        self.weight_decay = 5e-5

        self.center_loss_factor = 1e-2
        self.center_loss_alfa = 0.9

        self.model_def = 'models.inception_resnet_v1'
        self.filter_filename = ''
        self.filter_percentile=100.0
        self.filter_min_nrof_images_per_class=0
        self.embedding_size=128
        self.nrof_preprocess_threads=4
        self.learning_rate_schedule_file='data/learning_rate_schedule.txt'
        self.random_flip = True
        self.random_rotate = True
        self.random_crop = True
        self.log_histograms = False
        self.lfw_pairs='data/pairs.txt'
        self.lfw_file_ext='png'
        self.lfw_batch_size=100
        self.lfw_nrof_folds=10

        # Pre Train Model Make
        self.model_dir = train_softmax_main(self)
        self.output_file = self.model_dir+'/'+self.model_dir.replace(self.pre_model_dir,'')+'.pb'

        # make pb file
        freeze_graph_main(self)

if __name__ == '__main__':
    Facenet_run().run()