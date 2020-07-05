import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow.keras.models import load_model
import numpy as np
import cv2
from screeninfo import get_monitors
from time import sleep
from data_utils.face_utils import get_face_rect_dlib, crop_face_square, get_face_keypoints, get_eye_regions_from_shape

model = load_model('../trained_models/model_v2.h5')
model.summary()

monitors = get_monitors()
WIDTH = monitors[0].width
HEIGHT = monitors[0].height


def generate_random_target():
    black = np.zeros((HEIGHT, WIDTH, 3))
    random_point = np.random.random((2,))
    black[int(random_point[0] * HEIGHT) - 4:int(random_point[0] * HEIGHT) + 4,
    int(random_point[1] * WIDTH) - 4:int(random_point[1] * WIDTH) + 4] = (0, 255, 0)
    return black


def get_face(image):
    gray = tf.reduce_mean(image, -1)
    bool_map = tf.greater(gray, 25)
    row_bools = tf.reduce_any(bool_map, 1)
    col_bools = tf.reduce_any(bool_map, 0)
    rows = tf.where(row_bools)
    cols = tf.where(col_bools)
    row_min = rows[0][0]
    row_max = rows[-1][0]
    col_min = cols[0][0]
    col_max = cols[-1][0]
    rows, cols = image.shape[:2]
    t = row_min / rows
    b = row_max / rows
    l = col_min / cols
    r = col_max / cols
    new_image = tf.expand_dims(image[row_min:row_max, col_min:col_max], 0)
    new_image = tf.image.resize(new_image, (256, 256))[0]
    new_image = tf.cast(new_image, tf.uint8)
    return (l, t, r, b), new_image


def get_eyes(face):
    face_bbox = get_face_rect_dlib(face.numpy())
    if face_bbox is None:
        return None
    face_keypoints = get_face_keypoints(face.numpy(), face_bbox)
    retval = get_eye_regions_from_shape(face.numpy(), face_keypoints)
    if retval is None:
        return None
    leye, reye = retval
    return tf.image.convert_image_dtype(tf.convert_to_tensor(leye, dtype=tf.uint8),
                                        tf.float32), tf.image.convert_image_dtype(
        tf.convert_to_tensor(reye, dtype=tf.uint8), tf.float32)


def capture_face():
    cap = cv2.VideoCapture(0)
    sleep(0.5)
    ret, frame = cap.read()
    return frame


def to_input(image):
    (l, t, r, b), face = get_face(image)
    retval = get_eyes(face)
    if retval is None:
        return None
    leye, reye = retval
    face = tf.image.convert_image_dtype(face, tf.float32)
    return [tf.convert_to_tensor([face]), tf.convert_to_tensor([leye]), tf.convert_to_tensor([reye]),
            tf.convert_to_tensor([[l, t, r, b]], tf.float32)]


while True:
    img = generate_random_target()
    cv2.imshow('look here', img)
    cv2.namedWindow('look here', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('look here', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(0)
    face = capture_face()
    retval = get_face_rect_dlib(face)
    if retval is not None:
        (t, l), (b, r) = retval
        face = crop_face_square(face, ((t, l), (b, r)))
        print(face.shape)
        cv2.imshow('face', face)
        pred_xy = model.predict(to_input(face))[0]
        img[int(pred_xy[0] * HEIGHT) - 5:int(pred_xy[0] * HEIGHT) + 5,
        int(pred_xy[1] * WIDTH) - 5:int(pred_xy[1] * WIDTH) + 5, :] = (255, 0, 0)
        cv2.imshow('look here', img)
        cv2.namedWindow('look here', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('look here', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(0)
