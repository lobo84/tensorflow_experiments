import tensorflow as tf
import cv2
import numpy as np
import random

def loss(y_true, y_pred):
    return tf.reduce_mean((y_true[0]-y_pred[0])**2 + (y_true[1]-y_pred[1])**2)
    
def build_model(size, output_count):
    model = tf.keras.models.Sequential()
    padding='same'
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                                     activation='relu',padding=padding,
                                     input_shape=size))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',padding=padding))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu',padding=padding))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(output_count, activation='linear'))

    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])
    return model

def data(size):
    while True:
        x = random.randint(0, size[0])
        y = random.randint(0, size[1])
        radius = size[0] // 4
        img = np.zeros(size, np.uint8)
        cv2.circle(img, (x, y), radius, (0, 0, 255), cv2.FILLED)
        img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        t = np.array([float(x)/size[0],float(y)/size[1]])
        yield (img, t)
    
def batch_data(gen, size, batch_size):
    while True:
        batch_x = []
        batch_y = []
        for i in range(batch_size):
            x, y = next(gen)
            batch_x.append(x)
            batch_y.append(y)
        yield np.array(batch_x), np.array(batch_y)
    
def main():
    size = (64, 64, 3)
    output_count = 2
    datagen = data(size)
    model = build_model(size, output_count)
    model.summary()
    batch_gen = batch_data(datagen, size, 32)
    model.fit_generator(batch_gen, epochs=30, steps_per_epoch=10)

    eval_gen  = batch_data(datagen, size, 1)
    for i in range(50):
        x, _ = next(eval_gen)
        y = model.predict(x)
        print(y)
        cv2.circle(x[0], (int(y[0][0]*size[0]),int(y[0][1]*size[1])), 2, (255, 255, 255), cv2.FILLED)
        cv2.imshow('img', x[0])
        cv2.waitKey(0)
    cv2.destroyAllWindows()
        
#    model.evaluate(x_test, y_test)
    
    
if __name__ == "__main__":
    main()
