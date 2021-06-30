import tensorflow as tf #another way: from keras.datasets import fashion_mnist
import matplotlib
import matplotlib.pyplot as plt
from keras.utils import to_categorical

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



def load_Fashion_MNIST_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  
    return train_images, train_labels, test_images, test_labels


def explore_Fashion_MNIST_data(train_images, train_labels, test_images, test_labels):
    print("Train dataset shape:", train_images.shape)
    print("Train labels shape: ", train_labels)
    print("Sum of training images (also sum of train-labels): ", len(train_labels))
    
    #print(train_images)
    
    print("Test dataset shape:", test_images.shape)
    print("Test labels shape: ",test_labels)
    print("Sum of testing images (also sum of test-labels): ", len(test_labels))
    
    
    for i in range(5):
        show_Image(train_images[i], train_labels[i])
        

def show_Image(data, class_i):
    some_article = data   # Selecting the image.
    some_article_image = some_article.reshape(28, 28) # Reshaping it to get the 28x28 pixels
    plt.imshow(some_article_image, cmap = matplotlib.cm.binary, interpolation="nearest")
    #plt.imshow(some_article_image)
    #plt.axis("off")
    plt.colorbar()
    plt.grid(False)
    plt.xlabel(class_names[class_i])
    plt.show()
    
    
def prepare_dataset(train_images, train_labels, test_images, test_labels):
    nsamples, nx, ny = train_images.shape
    train_images = train_images.reshape((nsamples, nx, ny, 1))
    nsamples, nx, ny = test_images.shape
    test_images = test_images.reshape((nsamples, nx, ny, 1))
    
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return train_images, train_labels, test_images, test_labels


def scale_pixel_values(train_images, test_images):    
    train_images_scaled = train_images.astype('float32')
    test_images_scaled = test_images.astype('float32')
    train_images_scaled = train_images_scaled / 255.0
    test_images_scaled = test_images_scaled / 255.0
    return train_images_scaled, test_images_scaled

