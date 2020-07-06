import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from v3.model_v3 import get_model, get_loss
from v3.dataset_consume_v3 import get_dataset
import os

os.chdir('../')
train, valid = get_dataset()
train = train.batch(2)
model = get_model()
model.compile('adam', get_loss)
model.fit(train)
