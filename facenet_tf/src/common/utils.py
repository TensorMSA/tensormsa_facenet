import os, wget, bz2, dlib
import numpy as np
# common utils
def face_rotation_predictor_download(self):
    down_pred_url = self.down_land68_url
    bz_pred_file = self.land68_file
    down_pred_path = self.model_dir
    dt_pred_file = bz_pred_file.replace('.bz2', '')

    bz_pred_path = down_pred_path + bz_pred_file
    dt_pred_path = down_pred_path + dt_pred_file

    if not os.path.exists(down_pred_path):
        os.makedirs(down_pred_path)

    if os.path.isfile(bz_pred_path) == False:
        wget.download(down_pred_url, down_pred_path)
        zipfile = bz2.BZ2File(bz_pred_path)  # open the file
        data = zipfile.read()  # get the decompressed data
        newfilepath = bz_pred_path[:-4]  # assuming the filepath ends with .bz2
        open(newfilepath, 'wb').write(data)  # wr

    predictor = dlib.shape_predictor(dt_pred_path)
    detector = dlib.get_frontal_face_detector()
    return predictor, detector

def get_images_labels_pair(emb_array, labels):
    p = 0
    nrof_images = len(labels)
    emb_array_pair = np.zeros((nrof_images * nrof_images, emb_array.shape[1]))
    labels_pair = []
    for i in range(nrof_images):
        for j in range(nrof_images):
            embsub = np.subtract(emb_array[i], emb_array[j])
            emb_array_pair[p] = embsub
            p += 1

            if labels[i] == labels[j]:
                labels_pair.append(0)
            else:
                labels_pair.append(1)
    return emb_array_pair, labels_pair



