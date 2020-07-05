import os

import tensorflow as tf
from scipy.io import loadmat
from tqdm.auto import tqdm
from data_utils.face_utils import get_face_rect_dlib, get_face_keypoints, get_eye_regions_from_shape

DATA_ROOT = "data"
DEST_TFRECORDS_FOLDER = os.path.join(DATA_ROOT, "TFRecordsV2")


def get_screen_size(filepath):
    mat_loaded = loadmat(filepath)
    return mat_loaded['height_pixel'][0][0], mat_loaded['width_pixel'][0][0]


def to_features(face_loc, face, leye, reye, row_n, col_n):
    face_feature = to_jpeg_feature(face)
    leye_feature = to_jpeg_feature(leye)
    reye_feature = to_jpeg_feature(reye)
    pixel_feature = tf.train.Feature(
        float_list=tf.train.FloatList(
            value=[tf.convert_to_tensor(row_n, dtype=tf.float32), tf.convert_to_tensor(col_n, dtype=tf.float32)]))
    face_loc_feature = tf.train.Feature(
        float_list=tf.train.FloatList(
            value=[tf.cast(tf.convert_to_tensor(face_loc[0], dtype=tf.float64), dtype=tf.float32),
                   tf.cast(tf.convert_to_tensor(face_loc[1], dtype=tf.float64), dtype=tf.float32),
                   tf.cast(tf.convert_to_tensor(face_loc[2], dtype=tf.float64), dtype=tf.float32),
                   tf.cast(tf.convert_to_tensor(face_loc[3], dtype=tf.float64), dtype=tf.float32)]))

    example = {
        'face': face_feature,
        'leye': leye_feature,
        'reye': reye_feature,
        'pixel': pixel_feature,
        'face_loc': face_loc_feature
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=example))
    return example_proto.SerializeToString()


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
    return tf.convert_to_tensor(leye, dtype=tf.uint8), tf.convert_to_tensor(reye, dtype=tf.uint8)


def image_to_feature(image_filepath):
    image = tf.io.decode_image(tf.io.read_file(image_filepath))
    (l, t, r, b), face = get_face(image)
    retval = get_eyes(face)
    if retval is None:
        return None
    leye, reye = retval
    return (l, t, r, b), face, leye, reye


def to_jpeg_feature(image):
    image_jpeg = tf.io.encode_jpeg(image)
    image_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_jpeg.numpy()]))
    return image_feature


def download_dataset():
    PROCESSED = 0
    INVALID = 0
    if 'MPIIFaceGaze' not in os.listdir(DATA_ROOT):
        if 'MPIIFaceGaze.zip' not in os.listdir(DATA_ROOT):
            print("Downloading Dataset")
            os.system("wget -P {} http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip".format(DATA_ROOT))
        print("Extracting Dataset")
        os.system("unzip MPIIFaceGaze.zip -d {}".format(DATA_ROOT))
    print("Removing Previous TFRecords")
    os.system("rm -rf " + DEST_TFRECORDS_FOLDER)
    DATASET_ROOT = os.path.join(DATA_ROOT, 'MPIIFaceGaze')
    persons = os.listdir(DATASET_ROOT)
    persons.remove('readme.txt')
    os.makedirs(DEST_TFRECORDS_FOLDER)
    with tf.io.TFRecordWriter(os.path.join(DEST_TFRECORDS_FOLDER, 'FaceGazeValid.tfrecords')) as valid_writer:
        with tf.io.TFRecordWriter(os.path.join(DEST_TFRECORDS_FOLDER, 'FaceGazeTrain.tfrecords')) as train_writer:
            for PERSON in tqdm(persons):
                person_dir = os.path.join(DATASET_ROOT, PERSON)
                calibration_file = os.path.join(person_dir, 'Calibration', 'screenSize.mat')
                data_file = os.path.join(person_dir, PERSON + '.txt')
                height, width = get_screen_size(calibration_file)
                with open(data_file) as f:
                    for line in tqdm(list(f)):
                        PROCESSED += 1
                        parts = line.split(' ')
                        img_subdir = parts[0]
                        screen_col = int(parts[1])
                        screen_row = int(parts[2])
                        img_path = os.path.join(person_dir, img_subdir)
                        retval = image_to_feature(img_path)
                        if retval is None:
                            print("Skipped one!")
                            INVALID += 1
                            os.system('clear')
                            print("{}/{} are invalid".format(INVALID, PROCESSED))
                            continue
                        face_loc, img_jpeg, leye_jpeg, reye_jpeg = retval
                        feats = to_features(face_loc, img_jpeg, leye_jpeg, reye_jpeg, screen_row / height,
                                            screen_col / width)
                        if PERSON not in ['p00', 'p14']:
                            train_writer.write(feats)
                        else:
                            valid_writer.write(feats)


if __name__ == '__main__':
    download_dataset()
