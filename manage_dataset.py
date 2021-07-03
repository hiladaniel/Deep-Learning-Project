import tensorflow as tf #another way: from keras.datasets import fashion_mnist
import matplotlib
import matplotlib.pyplot as plt
from keras.utils import to_categorical

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_Fashion_MNIST_data():
    """
    This function import and load Fashion MNIST data directly from TensorFlow.   
    Returns 
    -------
    TYPE 
        DESCRIPTION
    train_images : numpy.ndarray
        NumPy array of grayscale images data with shapes (60000, 28, 28), containing the training data.
    train_labels : numpy.ndarray
        NumPy array of labels (integers in range 0-9) with shape (60000,) for the training data.
    test_images : numpy.ndarray
        NumPy array of grayscale images data with shapes (10000, 28, 28), containing the test data.
    test_labels : numpy.ndarray
        NumPy array of labels (integers in range 0-9) with shape (10000,) for the test data.
    """
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  
    return train_images, train_labels, test_images, test_labels


def explore_Fashion_MNIST_data(train_images, train_labels, test_images, test_labels):
    """    
    This function prints information about the four Fashion_MNIST NumPy arrays 
    and also displays the first five images of the training data.
    Parameters
    ----------
    TYPE 
        DESCRIPTION
    train_images : numpy.ndarray
        NumPy array of grayscale images data with shapes (60000, 28, 28), containing the training data.
    train_labels : numpy.ndarray
        NumPy array of labels (integers in range 0-9) with shape (60000,) for the training data.
    test_images : numpy.ndarray
        NumPy array of grayscale images data with shapes (10000, 28, 28), containing the test data.
    test_labels : numpy.ndarray
        NumPy array of labels (integers in range 0-9) with shape (10000,) for the test data.
    Returns
    -------
    None.
    """
    print("Train dataset shape:", train_images.shape)
    print("Train labels shape: ", train_labels)
    print("Sum of training images (also sum of train-labels): ", len(train_labels))
        
    print("Test dataset shape:", test_images.shape)
    print("Test labels shape: ",test_labels)
    print("Sum of testing images (also sum of test-labels): ", len(test_labels))
        
    for i in range(5): #Plot first five images of the training data.
        show_Image(train_images[i], train_labels[i])
        

def show_Image(data, class_i):
    """
    This function displays a single instance (image) of the input dataset (training data) on the screen.
    Parameters
    ----------
    TYPE 
        DESCRIPTION
    data : numpy.ndarray
        A single instance of the training data (an article image).
        The image is a NumPy array of grayscale image with shapes (28, 28).
    class_i : numpy.ndarray
        A label which associated with the image. 
        The label is an integer between 0 and 9, which represents the index of the sample's class.
    Returns
    -------
    None.
    """
    plt.imshow(data, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.colorbar()
    plt.grid(False)
    plt.xlabel(class_names[class_i]) #Plot the class-name of the item in the image.
    plt.show() #Show the figure.
    
    
def prepare_dataset(train_images, train_labels, test_images, test_labels):
    """
    This function does the first step of Data Preparation before training the network, 
    by reshaping the data arrays and transforming the labels' structure.    
    Parameters
    ----------
    TYPE 
        DESCRIPTION
    train_images : numpy.ndarray
        NumPy array of grayscale images data with shapes (60000, 28, 28), containing the training data.
    train_labels : numpy.ndarray
        NumPy array of labels (integers in range 0-9) with shape (60000,) for the training data.
    test_images : numpy.ndarray
        NumPy array of grayscale images data with shapes (10000, 28, 28), containing the test data.
    test_labels : numpy.ndarray
        NumPy array of labels (integers in range 0-9) with shape (10000,) for the test data.
    Returns
    -------
    train_images : numpy.ndarray
        NumPy array of grayscale images data with shapes (60000, 28, 28, 1), containing the training data.
    train_labels : numpy.ndarray
        NumPy array of labels with shape (60000, 10) for the training data.
    test_images : numpy.ndarray
        NumPy array of grayscale images data with shapes (10000, 28, 28, 1), containing the test data.
    test_labels : numpy.ndarray
        NumPy array of labels with shape (10000, 10) for the test data.
    """   
    #Reshape the data arrays to have a single color channel.
    nsamples, nx, ny = train_images.shape
    train_images = train_images.reshape((nsamples, nx, ny, 1))
    nsamples, nx, ny = test_images.shape
    test_images = test_images.reshape((nsamples, nx, ny, 1))
    #Using one hot encoding.
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return train_images, train_labels, test_images, test_labels


def scale_pixel_values(train_images, test_images): 
    """
    This function does the second step of Data Preparation before feeding it to the neural network model.
    The function normalizes the pixel values of the grayscale images in the dataset.    
    Parameters
    ----------
    TYPE 
        DESCRIPTION
    train_images : numpy.ndarray
        NumPy array of grayscale images data with shapes (60000, 28, 28, 1), containing the training data.
    test_images : numpy.ndarray
        NumPy array of grayscale images data with shapes (10000, 28, 28, 1), containing the test data.
    Returns
    -------
    train_images_scaled : numpy.ndarray
        NumPy array of grayscale images data after Feature Scaling with shapes (60000, 28, 28, 1), containing the training data.
    test_images_scaled : numpy.ndarray
        NumPy array of grayscale images data after Feature Scaling with shapes (10000, 28, 28, 1), containing the test data.
    """
    #Converting the data type from unsigned integers to floats.
    train_images_scaled = train_images.astype('float32')
    test_images_scaled = test_images.astype('float32')
    #Scale the pixel values of the images to a range of 0 to 1, by dividing them by the maximum value (255). 
    train_images_scaled = train_images_scaled / 255.0
    test_images_scaled = test_images_scaled / 255.0
    return train_images_scaled, test_images_scaled


