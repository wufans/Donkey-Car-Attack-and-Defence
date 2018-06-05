import os
import numpy as np
import keras

import donkeycar as dk


def default_categorical():
    from keras.layers import Input, Dense, merge
    from keras.models import Model
    from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Dense

    img_in = Input(shape=(120, 160, 3), name='img_in')                      # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)       # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)       # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu')(x)       # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)       # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 1wx1h stride, relu

    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    x = Flatten(name='flattened')(x)                                        # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)                                    # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)                                                      # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)                                     # Classify the data into 50 features, make all negatives 0
    x = Dropout(.1)(x)                                                      # Randomly drop out 10% of the neurons (Prevent overfitting)
    #categorical output of the angle
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)        # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0

    #continous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)      # Reduce to 1 number, Positive number only

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'categorical_crossentropy',
                        'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': .001})

    return model

def train(self, train_gen, val_gen,
          saved_model_path, epochs=100, steps=100, train_split=0.8,
          verbose=1, min_delta=.0005, patience=5, use_early_stop=True):

    """
    train_gen: generator that yields an array of images

    """

    #checkpoint to save model after each epoch
    # 1. filename：字符串，保存模型的路径
    # 2. monitor：需要监视的值
    # 3. verbose：信息展示模式，0或1
    # 4. save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
    # 5. mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当检测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
    # 6. save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
    # 7. period：CheckPoint之间的间隔的epoch数
    save_best = keras.callbacks.ModelCheckpoint(saved_model_path,
                                                monitor='val_loss',
                                                verbose=verbose,
                                                save_best_only=True,
                                                mode='min')

    #stop training if the validation error stops improving.
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=min_delta,
                                               patience=patience,
                                               verbose=verbose,
                                               mode='auto')

    callbacks_list = [save_best]

    if use_early_stop:
        callbacks_list.append(early_stop)

    self.model = default_linear()

    hist = self.model.fit_generator(
                    train_gen,
                    steps_per_epoch=steps,
                    epochs=epochs,
                    verbose=1,
                    validation_data=val_gen,
                    callbacks=callbacks_list,
                    validation_steps=steps*(1.0 - train_split)/train_split)
    return hist
## main part
def my_train(cfg, tub_names, model_name):
    '''
    use the specified data in tub_names to train an artifical neural network
    saves the output trained model as model_name
    '''
    X_keys = ['cam/image_array']
    y_keys = ['user/angle', 'user/throttle']

    def rt(record):
        record['user/angle'] = dk.utils.linear_bin(record['user/angle'])
        return record

    kl = KerasCategorical()
    print('tub_names', tub_names)
    if not tub_names:
        tub_names = os.path.join(cfg.DATA_PATH, '*')
    tubgroup = TubGroup(tub_names)
    train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys, record_transform=rt,
                                                    batch_size=cfg.BATCH_SIZE,
                                                    train_frac=cfg.TRAIN_TEST_SPLIT)

    model_path = os.path.expanduser(model_name)

    total_records = len(tubgroup.df)
    total_train = int(total_records * cfg.TRAIN_TEST_SPLIT)
    total_val = total_records - total_train
    print('train: %d, validation: %d' % (total_train, total_val))
    steps_per_epoch = total_train // cfg.BATCH_SIZE
    print('steps_per_epoch', steps_per_epoch)

    kl.train(train_gen,
             val_gen,
             saved_model_path=model_path,
             steps=steps_per_epoch,
             train_split=cfg.TRAIN_TEST_SPLIT)




model = "mytestmodel"
tub = "data"
cfg = dk.load_config()

train(cfg, tub, model)
