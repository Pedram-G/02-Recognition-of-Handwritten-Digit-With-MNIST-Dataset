from keras.models import Model
from keras import layers
import keras

def Create_Models():
    
    # Model Case1 
    myInput = layers.Input(shape=(28,28,1))
    conv1 = layers.Conv2D(filters = 32, kernel_size= 3, activation = 'relu')(myInput)
    conv2 = layers.Conv2D(filters = 64, kernel_size= 3, activation = 'relu')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2,2))(conv2)
    drop1 = layers.Dropout(rate=0.25)(pool1)
    flat1 = layers.Flatten()(drop1)
    den1 = layers.Dense(units = 128, activation = 'relu')(flat1)
    drop2 = layers.Dropout(rate=0.5)(den1)
    out_layer = layers.Dense(10, activation='softmax')(drop2)

    Model_Case1 = Model(myInput, out_layer)

    Model_Case1.summary()
    Model_Case1.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    # Model Case2 
    myInput = layers.Input(shape=(28,28,1))
    conv1 = layers.Conv2D(filters = 32, kernel_size= 3, activation = 'relu')(myInput)
    pool1 = layers.MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = layers.Conv2D(filters = 64, kernel_size= 3, activation = 'relu')(pool1)
    pool2 = layers.MaxPooling2D(pool_size=(2,2))(conv2)
    drop1 = layers.Dropout(rate=0.25)(pool2)
    flat1 = layers.Flatten()(drop1)
    den1 = layers.Dense(units = 128, activation = 'relu')(flat1)
    drop2 = layers.Dropout(rate=0.5)(den1)
    out_layer = layers.Dense(10, activation='softmax')(drop2)

    Model_Case2 = Model(myInput, out_layer)

    Model_Case2.summary()
    Model_Case2.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    # Model Case3 
    myInput = layers.Input(shape=(28,28,1))
    conv1 = layers.Conv2D(filters = 32, kernel_size= 3, activation = 'relu')(myInput)
    conv2 = layers.Conv2D(filters = 64, kernel_size= 3, activation = 'relu')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2,2))(conv2)
    flat1 = layers.Flatten()(pool1)
    den1 = layers.Dense(units = 128, activation = 'relu')(flat1)
    out_layer = layers.Dense(10, activation='softmax')(den1)

    Model_Case3 = Model(myInput, out_layer)

    Model_Case3.summary()
    Model_Case3.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    # Model Case4 
    myInput = layers.Input(shape=(28,28,1))
    conv1 = layers.Conv2D(filters = 32, kernel_size= 3, activation = 'relu')(myInput)
    pool1 = layers.MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = layers.Conv2D(filters = 64, kernel_size= 3, activation = 'relu')(pool1)
    pool2 = layers.MaxPooling2D(pool_size=(2,2))(conv2)
    flat1 = layers.Flatten()(pool2)
    den1 = layers.Dense(units = 128, activation = 'relu')(flat1)
    out_layer = layers.Dense(10, activation='softmax')(den1)

    Model_Case4 = Model(myInput, out_layer)

    Model_Case4.summary()
    Model_Case4.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    # Model Case5 
    myInput = layers.Input(shape=(28,28,1))
    conv1 = layers.Conv2D(filters = 32, kernel_size= 3, activation = 'relu')(myInput)
    conv2 = layers.Conv2D(filters = 64, kernel_size= 3, activation = 'relu')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2,2))(conv2)
    flat1 = layers.Flatten()(pool1)
    out_layer = layers.Dense(10, activation='softmax')(flat1)

    Model_Case5 = Model(myInput, out_layer)

    Model_Case5.summary()
    Model_Case5.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    # Model Case6 
    myInput = layers.Input(shape=(28,28,1))
    conv1 = layers.Conv2D(filters = 32, kernel_size= 3, activation = 'relu')(myInput)
    pool1 = layers.MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = layers.Conv2D(filters = 64, kernel_size= 3, activation = 'relu')(pool1)
    pool2 = layers.MaxPooling2D(pool_size=(2,2))(conv2)
    flat1 = layers.Flatten()(pool2)
    den1 = layers.Dense(units = 128, activation = 'relu')(flat1)
    drop1 = layers.Dropout(rate=0.25)(den1)
    out_layer = layers.Dense(10, activation='softmax')(drop1)

    Model_Case6 = Model(myInput, out_layer)

    Model_Case6.summary()
    Model_Case6.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    
    return(Model_Case1, Model_Case2, Model_Case3, Model_Case4, Model_Case5, Model_Case6)
