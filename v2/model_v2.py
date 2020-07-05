import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Reshape, Dense, Concatenate, Dropout

os.system('clear')


def get_model():
    face_input = Input((256, 256, 3), name='face')

    face_conv1 = Conv2D(8, (7, 7), activation='relu', )(face_input)
    face_conv1 = BatchNormalization()(face_conv1)
    face_maxpool1 = MaxPool2D()(face_conv1)

    face_conv2 = Conv2D(16, (5, 5), activation='relu')(face_maxpool1)
    face_conv2 = BatchNormalization()(face_conv2)
    face_maxpool2 = MaxPool2D()(face_conv2)

    face_conv3 = Conv2D(32, (5, 5), activation='relu')(face_maxpool2)
    face_conv3 = BatchNormalization()(face_conv3)
    face_maxpool3 = MaxPool2D()(face_conv3)

    face_conv4 = Conv2D(64, (3, 3), activation='relu')(face_maxpool3)
    face_conv4 = BatchNormalization()(face_conv4)
    face_maxpool4 = MaxPool2D()(face_conv4)

    face_feats_shape = face_maxpool4.shape
    face_to1x1 = Conv2D(128, (face_feats_shape[1:3]), activation='relu')(face_maxpool4)
    face_for_concat = Reshape((128,))(face_to1x1)

    leye_input = Input((64, 64, 3), name='leye')
    leye_conv1 = Conv2D(8, (5, 5), activation='relu')(leye_input)
    leye_conv1 = BatchNormalization()(leye_conv1)
    leye_maxpool1 = MaxPool2D()(leye_conv1)

    leye_conv2 = Conv2D(32, (3, 3), activation='relu')(leye_maxpool1)
    leye_conv2 = BatchNormalization()(leye_conv2)
    leye_maxpool2 = MaxPool2D()(leye_conv2)

    leye_feats_shape = leye_maxpool2.shape
    leye_to1x1 = Conv2D(64, leye_feats_shape[1:3])(leye_maxpool2)
    leye_for_concat = Reshape((64,))(leye_to1x1)

    reye_input = Input((64, 64, 3), name='reye')
    reye_conv1 = Conv2D(8, (5, 5), activation='relu')(reye_input)
    reye_conv1 = BatchNormalization()(reye_conv1)
    reye_maxpool1 = MaxPool2D()(reye_conv1)

    reye_conv2 = Conv2D(32, (3, 3), activation='relu')(reye_maxpool1)
    reye_conv2 = BatchNormalization()(reye_conv2)
    reye_maxpool2 = MaxPool2D()(reye_conv2)

    reye_feats_shape = reye_maxpool2.shape
    reye_to1x1 = Conv2D(64, reye_feats_shape[1:3])(reye_maxpool2)
    reye_for_concat = Reshape((64,))(reye_to1x1)

    face_loc_input = Input((4,), name='face_loc')
    face_loc_for_concat = Dense(64, activation='relu')(face_loc_input)

    concat = Concatenate()([face_for_concat, leye_for_concat, reye_for_concat, face_loc_for_concat])
    all_batch_norm = BatchNormalization()(concat)
    all_batch_norm = Dropout(0.25)(all_batch_norm)
    all_dense_1 = Dense(128, activation='relu', activity_regularizer='l2')(all_batch_norm)
    all_dense_1 = BatchNormalization()(all_dense_1)
    all_dense_2 = Dense(32, activation='relu', activity_regularizer='l2')(all_dense_1)
    all_dense_2 = BatchNormalization()(all_dense_2)
    head = Dense(2, activation='sigmoid')(all_dense_2)
    model = Model(inputs=[face_input, leye_input, reye_input, face_loc_input], outputs=head)
    return model


if __name__ == '__main__':
    get_model()
