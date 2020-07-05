# GPUU Bugfix
import tensorflow

physical_devices = tensorflow.config.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow.keras.models import load_model
from v2.dataset_consume_v2 import get_dataset
import numpy as np
import cv2

model = load_model('../trained_models/model_v2.h5')
model.summary()
train, valid = get_dataset()
valid = valid.batch(1)
for ips, xy in valid:
    black_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.imshow('eye_image', ips[0][0].numpy())
    pred_xy = model.predict(ips)[0]
    xy = xy[0].numpy()
    black_img[int(pred_xy[0] * 720) - 5:int(pred_xy[0] * 720) + 5,
    int(pred_xy[1] * 1280) - 5:int(pred_xy[1] * 1280) + 5, :] = (255, 0, 0)
    black_img[int(xy[0] * 720) - 5:int(xy[0] * 720) + 5, int(xy[1] * 1280) - 5:int(xy[1] * 1280) + 5, :] = (0, 255, 0)
    cv2.imshow('predictions', black_img)
    cv2.waitKey(0)
