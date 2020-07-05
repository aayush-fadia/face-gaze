import tensorflow as tf
import os
import cv2

os.system('clear')


def parse_example(example):
    feat_spec = {
        'face': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'leye': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'reye': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'pixel': tf.io.FixedLenFeature([2], tf.float32, default_value=(0, 0)),
        'face_loc': tf.io.FixedLenFeature([4], tf.float32, default_value=(0, 0, 0, 0)),
    }
    return tf.io.parse_single_example(example, feat_spec)


def get_image(image_jpeg):
    image = tf.io.decode_jpeg(image_jpeg)
    image = tf.cast(image, tf.float32)
    image = image / 255.
    return image


def get_XY(example):
    parsed = parse_example(example)
    face = get_image(parsed['face'])
    leye = get_image(parsed['leye'])
    reye = get_image(parsed['reye'])
    face_loc = parsed['face_loc']
    return (face, leye, reye, face_loc), parsed['pixel']


def visualize(x, y):
    x = x * 255.
    x = tf.cast(x, tf.uint8)
    x = x.numpy()
    cv2.imshow('x', x)
    cv2.waitKey(0)


ds_train = tf.data.TFRecordDataset("TFRecordsV2/FaceGazeTrain.tfrecords")
ds_train = ds_train.map(get_XY)

ds_valid = tf.data.TFRecordDataset("TFRecordsV2/FaceGazeValid.tfrecords")
ds_valid = ds_valid.map(get_XY)


def get_dataset():
    return ds_train, ds_valid


if __name__ == "__main__":
    train, valid = get_dataset()
    for item in valid.take(4):
        print(len(item[0]))
