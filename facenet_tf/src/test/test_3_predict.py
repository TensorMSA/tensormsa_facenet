from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from facenet_tf.src.run_9_predict import FaceRecognitionRun

if __name__ == '__main__':
    print('==================================================')
    FaceRecognitionRun().realtime_run('test')