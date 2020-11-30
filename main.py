# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# import panda as pd
import numpy as np
import check_resources as check
import frontalize as frontalize
import matplotlib.pyplot as plt
import facial_feature_detector as feature_detection
import camera_calibration as calib
import scipy.io as io
import os
import cv2
import keras
import tensorflow as tf
import models
from keras import losses
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

batch_size = 128
img_height = 48
img_width = 48

this_path = os.path.dirname(os.path.abspath(__file__))
data_dir = "data/train"
front_dir = "data/front"

def train():
    # training dataset
    train_ds = keras.preprocessing.image_dataset_from_directory(
        front_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # validation dataset
    val_ds = keras.preprocessing.image_dataset_from_directory(
        front_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    # visualizing the data
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")
    # plt.show()


    # configuration for better performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # base model
    base_model = models.base_model()
    base_model.compile(optimizer='adam',
                       loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])
    history = base_model.fit(train_ds,
                             validation_data=val_ds,
                             epochs=10)


def face_frontalization():
    dir = "data/train"
    # check for dlib saved weights for face landmark detection
    # if it fails, dowload and extract it manually from
    # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
    check.check_dlib_landmark_weights()

    # load detections performed by dlib library on 3D model and Reference Image
    model3D = frontalize.ThreeD_Model(this_path + "/frontalization_models/model3Ddlib.mat", 'model_dlib')

    # load mask to exclude eyes from symmetry
    eyemask = np.asarray(io.loadmat('frontalization_models/eyemask.mat')['eyemask'])
    default_dir = os.getcwd()
    for c in os.listdir(dir):
        if os.path.isdir(dir + "/" + str(c)):
            os.mkdir("data/front/" + c)
            for f in os.listdir(dir + "/" + str(c)):
                if f.endswith("jpg"):
                    img = cv2.imread(dir + "/" + str(c) + "/" + str(f), 1)
                    # plt.title('Query Image')
                    # plt.imshow(img[:, :, ::-1])
                    # plt.show()
                    # cast img to type int for cv2

                    img = img.astype(np.uint8)
                    # create a color version for frontalizer stuffs

                    # extract landmarks from the query image
                    # list containing a 2D array with points (x, y) for each face detected in the query image
                    lmarks = feature_detection.get_landmarks(img)
                    if type(lmarks) is np.ndarray and len(lmarks) > 0:
                        # perform camera calibration according to the first face detected
                        proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])

                        # perform frontalization and convert result to grayscale
                        frontal_raw, frontal_sym = frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)
                        temp = cv2.cvtColor(frontal_raw, cv2.COLOR_BGR2GRAY)

                        # find nonzero bbox and crop image to remove uncessesary black space from edges
                        temp_mask = cv2.findNonZero(temp)
                        t_x, t_y, t_w, t_h = cv2.boundingRect(temp_mask)
                        t_bbox = temp[t_y:t_y + t_h, t_x:t_x + t_w]

                        # resize the cropped image to the appropriate dimensions for network
                        t_bbox = cv2.resize(t_bbox, dsize=(48, 48))
                        t_bbox = np.resize(t_bbox, (48, 48, 1))

                        os.chdir("data/front/" + c)
                        cv2.imwrite("front_" + f, t_bbox)
                        os.chdir(default_dir)
                        # plt.imshow(t_bbox[:,:,::-1])
                        # plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # face_frontalization()
    train()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
