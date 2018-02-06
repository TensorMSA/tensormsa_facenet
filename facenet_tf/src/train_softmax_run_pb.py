from facenet_tf import init_value

from facenet_tf.src.common.freeze_graph import main as freeze_graph_main

class Facenet_run():
    def run(self):
        init_value.init_value.init(self)

        self.model_dir = self.model_dir + self.pre_model_name + '/'
        self.output_file = self.model
        freeze_graph_main(self)

if __name__ == '__main__':
    Facenet_run().run()