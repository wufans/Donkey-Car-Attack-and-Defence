import tensorflow as tf
import keras

from donkeycar.parts.keras import KerasCategorical


def fgm(model, x_image):
    """
    Fast gradient method to generate adversarial examples of donkeycar

    """
    x_adv_image = tf.identity(x_image) #返回相同tensor的op


    return x_adv_image



def default_model():
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

def train(sess, env, data_path, model_path, shuffle=True, load=False,
          batch_size = 32,model_name = "pilot"):
    if load:
        if not hasattr(env, 'saver'):
            return print("Eroor: cannot find saver op")
        print('Loading saved model...')
        return env.saver.restore(sess,'model/{}'.format(name))

    print("Training model...")
    model = default_model()
    img_arr = img_arr.reshape((1,) + img_arr.shape)
    angle_binned, throttle = self.model.predict(img_arr)
    #print('throttle', throttle)
    #angle_certainty = max(angle_binned[0])
    angle_unbinned = dk.utils.linear_unbin(angle_binned)
    return angle_unbinned, throttle[0][0]

if __name__ == '__main__':

    args = docopt(__doc__)
    # 模型的路径
    model_path = "model_path"
    # 存放图片数据的路径
    x_path = "x_data_path"

    if args['model_path']:
        model_path = args['--model']
    if args['x_path']:
        x_path = args['--x_path']

    kl = KerasCategorical()
    model = kl.load(model_path)
    x_adv = fgm(model,x_image)

    print("Train a autopilot model")
    train()
