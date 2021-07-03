from model import define_model
from model import summarize_model
from model import visualize_model

from manage_dataset import load_Fashion_MNIST_data
from manage_dataset import explore_Fashion_MNIST_data
from manage_dataset import show_Image
from manage_dataset import prepare_dataset
from manage_dataset import scale_pixel_values

from keras.models import load_model

import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from numpy import mean
from numpy import std


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def training_and_validation(train_images, train_labels, n_folds=5):
    dataX = train_images
    dataY = train_labels
    acc_scores, histories = list(), list() 
    #acc_scores - the classification accuracy scores which collected during each fold of the k-fold cross-validation.    
	#models_histories - collected training histories
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
    print("\nClassification accuracy for each fold of the cross-validation process: ") 
    for train_img_ind, test_img_ind in kfold.split(dataX):
		# define model
        model = define_model()
		# select rows for train and test
        trainX, trainY, testX, testY = dataX[train_img_ind], dataY[train_img_ind], dataX[test_img_ind], dataY[test_img_ind]
		# fit model
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        model.save('final_model.h5')
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # append scores
        acc_scores.append(acc)
        histories.append(history)
    return acc_scores, histories, model


def plot_accuracy(histories):
    for i in range(len(histories)): 
        #plt.subplot(212) 
        plt.title('Training and Validation accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train') 
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')        
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['training accuracy', 'test accuracy'], loc='lower right')    
    plt.show()


def plot_loss(histories):
    for i in range(len(histories)):
        #plt.subplot(211) 
        plt.title('Training and Validation loss')  
        plt.plot(histories[i].history['loss'], color='blue', label='train') 
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')        
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['training loss', 'test loss'], loc='lower right')
    plt.show()
    
 
def plot_accuracy_scores(scores):
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	plt.boxplot(scores)
	plt.show()


def print_menu():
    print("\nWhat would you like to do?")
    print("1. Get to know the dataset")
    print("2. Train the model")
    print("3. Load the model")
    print("4. Test the model")
    print("5. Predict an image")
    print("6. Exit")
    choice = input("Enter your choice: ")
    while(choice.isdigit()==False or (int(choice) < 1) or (int(choice) > 6)):
        print("Sorry, this is not a valid input. ")
        choice = input("Please enter your choice again: ")
    return int(choice)


def option_1(train_images_orl, train_labels_orl, test_images_orl, test_labels_orl):
    explore_Fashion_MNIST_data(train_images_orl, train_labels_orl, test_images_orl, test_labels_orl)


def option_2(train_images, train_labels):
    print("What do you prefer?")
    print("1. Doing training and validation. \n2. Doing training only.")
    spec_choice = input("Enter your choice: ")   
    while(spec_choice.isdigit()==False or int(spec_choice) != 1 and int(spec_choice) != 2):
        print("Sorry, this is not a valid input.")
        spec_choice = input("Please enter your choice again: ")
    if (int(spec_choice)==1):
        scores, histories, model = training_and_validation(train_images, train_labels)
        summarize_model(model)
        visualize_model(model)
        #plot the diagnostics of the learning behavior of the model during training
        plot_accuracy(histories)
        plot_loss(histories)
        #plot the estimation of the model performance
        plot_accuracy_scores(scores)  
    else:
        model = define_model()
        summarize_model(model)
        visualize_model(model)
        model.fit(train_images, train_labels, epochs = 10, batch_size = 32, verbose = 0)
        model.save('final_model.h5')
        print("Done! \nThe model is successfully trained and saved.")
    return True, model


def option_3():
    model = load_model('final_model.h5')
    print("Done. Model is loaded.") 
    return True, model


def option_4(test_images, test_labels, model):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0) 
    #print("\nTest accuracy: ", test_acc * 100.0)
    print("\nTest accuracy: ", test_acc)
    print("Test loss: ", test_loss)
    

def option_5(test_images_orl, test_labels_orl, model):
    img_num = input("Choose a number between 1 to 10,000: ")
    i = int(img_num) - 1 #index of the chosen image
    show_Image(test_images_orl[i], test_labels_orl[i])
    print("You choose to predict ", class_names[test_labels_orl[i]])
    
    img = test_images_orl[i]
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img / 255.0
    
    result = model.predict_classes(img)
    print("Prediction is: ",class_names[result[0]])
    #print(result[0], test_labels_orl[i])
    if (result[0] == test_labels_orl[i]):
        print("Prediction is true :)")
    else:
        print("Prediction is wrong :(")


def main():
    X_train, y_train, X_test, y_test = load_Fashion_MNIST_data()
    X_train_orl, y_train_orl, X_test_orl, y_test_orl = X_train, y_train, X_test, y_test
    X_train, y_train, X_test, y_test = prepare_dataset( X_train, y_train, X_test, y_test)    
    X_train, X_test = scale_pixel_values(X_train, X_test)
    
    print("\nWellcome! \nThis is a deep learning project of Clothes Classifier using Fashion MNIST Dataset.")
    user_choice = print_menu()
    model_loaded = False
    while not user_choice == 6:
        if user_choice == 1:
            option_1(X_train_orl, y_train_orl, X_test_orl, y_test_orl)
        elif user_choice == 2:
            model_loaded, model = option_2(X_train, y_train)            
        elif user_choice == 3:
            model_loaded, model = option_3() 
        elif user_choice == 4:
            if not model_loaded:
                print("Please load or train the model first.")
            else:
                option_4(X_test, y_test, model)
        else:
            #user_choise = 5
            if not model_loaded:
                print("Please load or train the model first.")
            else:
                option_5(X_test_orl, y_test_orl, model)    
        user_choice = print_menu()
   

if __name__ == "__main__":
    main()   
