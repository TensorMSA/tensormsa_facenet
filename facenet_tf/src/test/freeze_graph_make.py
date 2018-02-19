from facenet_tf import init_value
from facenet_tf.src.common.train_softmax import main as train_softmax_main
from facenet_tf.src.common.freeze_graph import main as freeze_graph_main

class Facenet_run():
    def run(self):
        init_value.init_value.init(self)

        # Pre Train Model Make
        self.model_name = '20180214-075119'
        self.model_dir = '/home/dev/tensormsa_facenet/facenet_tf/pre_model/'+self.model_name+'/'
        self.output_file = self.model_dir+self.model_name+'.pb'
        # '/home/dev/tensormsa_facenet/facenet_tf/pre_model/20180212-062537/20180212-062537.pb'

        # make pb file
        freeze_graph_main(self)

if __name__ == '__main__':
    Facenet_run().run()