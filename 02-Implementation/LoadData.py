from keras.datasets import mnist
from keras.utils import np_utils

def Load_Data():
    
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    X_train = train_images.reshape(60000, 28, 28, 1)
    X_test = test_images.reshape(10000, 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    Y_train = np_utils.to_categorical(train_labels)
    Y_test = np_utils.to_categorical(test_labels)
    
    return(X_train,X_test,Y_train,Y_test)









