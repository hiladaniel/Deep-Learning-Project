from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model


def define_model():
    """
    This function defines a convolutional neural network model for the problem.
    Returns
    -------
    TYPE 
        DESCRIPTION
    model : tensorflow.python.keras.engine.sequential.Sequential
        A CNN model for the problem.
    """
    model = Sequential() 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1))) 
    model.add(MaxPooling2D((2, 2))) 
    model.add(Flatten()) 
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform')) 
    model.add(Dense(10, activation='softmax'))
	#Compile the model. 
    opt = SGD(lr=0.01, momentum=0.9) 
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']) 
    return model 


def summarize_model(model):
    print(model.summary())

    
def visualize_model(model):
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)  
    print("\nA plot of the neural network model is waiting for you in the folder where project.py is saved!")
    
    
    