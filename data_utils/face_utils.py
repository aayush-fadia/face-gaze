import cv2
import dlib
import numpy as np
from imutils import face_utils

predictor = dlib.shape_predictor('../shape_predictor_68_face_landmarks.dat')

detector = dlib.get_frontal_face_detector()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def get_face_rect_dlib(frame):
    """
    Get face rectangle using DLib's face detector, on a CLAHE enhanced image.
    :param frame: RGB frame from the webcam
    :return: Two Tuples (points) (top, left), (bottom, right)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    rects = detector(gray, 0)
    return ((rects[0].top(), rects[0].left()), (rects[0].bottom(), rects[0].right())) if len(rects) > 0 else None


def draw_face_rect_dlib(frame, face_rect):
    if face_rect is not None:
        (t, l), (b, r) = face_rect
        cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0))


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def get_face_keypoints(frame, face_rect):
    (t, l), (r, b) = face_rect
    dlib_rect = dlib.rectangle(l, t, b, r)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_shape = predictor(frame, dlib_rect)
    face_shape = face_utils.shape_to_np(face_shape)
    return face_shape


def get_eye_regions_from_shape(image, face_shape):
    both_eyes = face_shape[18:30]
    min_xy = np.min(both_eyes, 0)
    max_xy = np.max(both_eyes, 0)
    minx, maxx, miny, maxy = min_xy[0], max_xy[0], min_xy[1], max_xy[1]
    eye_region = image[miny:maxy, minx:maxx, :]
    _, width, _ = eye_region.shape
    width = int(width / 2)
    try:
        leye = cv2.resize(eye_region[:, :width], (64, 64))
        reye = cv2.resize(eye_region[:, width:], (64, 64))
        return leye, reye
    except Exception:
        return None
