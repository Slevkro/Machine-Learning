import numpy as np 
from matplotlib import image
from matplotlib import pyplot
import os 
import copy

from PIL import Image
import PIL
"""
image = Image.open('/media/brandonC/USB DRIVE/archive/sad/46.jpg').resize((256, 256))

data = np.asarray(image)
print(type(data))
print(data.shape)


print(im.format)
print(im.size)
print(im.mode)
print('==============')
print(im_resized.format)
print(im_resized.size)
print(im_resized.mode)

path, dirs, files = next(os.walk("/media/brandonC/USB DRIVE/archive/sad/"))
print(len(files))
print(files)

pyplot.imshow(image)
pyplot.show()
"""

def load_data(path, classes, num_px):
    images_test_array = np.zeros((num_px * num_px * 3 + 1, 1), dtype=float)
    images_train_array = np.zeros((num_px * num_px * 3 + 1, 1), dtype=float)
    
    for i in range(len(classes)):
        path_class = path+classes[i]
        path_dir, dirs, files = next(os.walk(path_class))

        size_train_set = round(len(files) * 0.7)
        size_test_set = len(files) - size_train_set
        #print(size_test_set)
        #print(size_train_set)
        count_samples = 0

        for image_name in files:
            count_samples += 1
            image = Image.open(path_class + "/" + image_name).resize((num_px, num_px)).convert("RGB")
            #data = np.asarray(image)
            #print(data.shape)
            image_data = np.asarray(image).reshape(num_px * num_px * 3, 1)
            # Paste labels on top
            image_data = np.concatenate([np.full((1,1), i), image_data])

            if count_samples <= size_train_set:
                images_train_array = np.concatenate([images_train_array, image_data], axis = 1)
            else:
                images_test_array = np.concatenate([images_test_array, image_data], axis = 1)
        
    #Delete initial zeros
    images_train_array = np.delete(images_train_array, 0, 1)
    images_test_array = np.delete(images_test_array, 0, 1)
    #We shuffle data in columns with T
    np.random.shuffle(images_train_array.T)
    np.random.shuffle(images_test_array.T)

    #Substract lables
    train_labels = images_train_array[0,:].reshape(1,images_train_array.shape[1])
    test_labels = images_test_array[0,:].reshape(1,images_test_array.shape[1])
    #train_labels = np.split(images_train_array, images_train_array.shape[0])[0]
    #test_labels = np.split(images_test_array, images_test_array.shape[0])[0]

    # We delete labels from sets
    images_train_array = np.delete(images_train_array, 0, 0)
    images_test_array = np.delete(images_test_array, 0, 0)


    return train_labels, images_train_array/255, test_labels, images_test_array/255

# Here goes your absolute path until all folders
sets_root_path = "/media/brandonC/USB DRIVE/archive/"

# Here goes your clasifiers
classifiers = ["shock", "no_shock"]

Y_train, X_train, Y_test, X_test = load_data(sets_root_path, classifiers, num_px = 256)
print("Train shape: " + str(X_train.shape))
print("Train labels shape: " + str(Y_train.shape))
print("Test shape: " + str(X_test.shape))
print("Test labels shape: " + str(Y_test.shape))
print(X_train)


mona_china = np.hsplit(X_train, X_train.shape[1])[0].reshape(256,256,3)

#mona_china = np.asarray(Image.open('/media/brandonC/USB DRIVE/archive/happy/004_158_367_212_212.png').resize((256, 256)).convert("RGB"))
print(mona_china.shape)
#print(mona_china)
print("De la clase: " + str(Y_train[0,0]))



pyplot.imshow(mona_china)
pyplot.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))
"""
print("Sigmoid [-1 0 1]" + str(sigmoid(np.array([-1, 0, 1]))))
"""
def inicialize_with_zeros(dim):
    #dim = x.shape[0]
    #w.shape === (salidas, entradas)
    #b.shape === (salidas, 1)

    w = np.zeros((dim, 1))
    b = 0.0 
    return w, b 

"""
W, B = inicialize_with_zeros(2)

print(W)
print(B)
"""

def propagate(w, b, X, Y):
    #Number of training examples
    m = X.shape[1]
    #Value of activation function for all examples on X
    A = sigmoid(np.dot(w.T, X) + b)

    #Calculate cost function that includes all training examples
    epsilon = 1e-5
    cost = - (1 / m) * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))

    #Start calculate backward prop
    #Gradient for the cost function on the point with the current parameters
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    

    #print(type(cost))

    cost = np.squeeze(np.array(cost))

    #print(type(cost))

    grads = {"dw": dw,
            "db": db}
    
    return grads, cost

"""
w =  np.array([[1.], [2.]])
b = 2.
X =np.array([[1., 2., -1.], [3., 4., -3.2]])
Y = np.array([[1, 0, 1]])
grads, cost = propagate(w, b, X, Y)

print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
""" 

def gradient_decent(w, b, X, Y, num_iterations = 100, learning_rate = 0.009, print_cost = False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
        #Calculate the firs forward prop
        grads, cost = propagate(w, b, X, Y)

        #Gradients with those parametters
        dw = grads["dw"]
        db = grads["db"]

        #Update parammeters acording with grads
        w = w - learning_rate * dw 
        b = b - learning_rate * db 

        #Record of costs 
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("Cost after iteration %i %f" %(i, cost))
        
    params = {"w": w,
            "b": b}
    grads = {"dw": dw,
            "db": db}
    
    return params, grads, costs
"""
params, grads, costs = gradient_decent(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print("Costs = " + str(costs))
"""

def output_predict(w, b, X):

    A = sigmoid(np.dot(w.T, X) + b)

    A[A>0.5] = 1
    A[A<=0.5] = 0

    return A
"""
w = np.array([[0.1124579], [0.23106775]])
b = -0.3
X = np.array([[1., -1.1, -3.2],[1.2, 2., 0.1]])
print ("predictions = " + str(output_predict(w, b, X)))
"""
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = inicialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = gradient_decent(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = output_predict(w, b, X_test)
    Y_prediction_train = output_predict(w, b, X_train)

    #Print train and test errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    data_model = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return data_model

logistic_regression_model = model(X_train, Y_train, X_test, Y_test, num_iterations=1000, learning_rate=0.001, print_cost=True)

# Plot learning curve (with costs)
costs = np.squeeze(logistic_regression_model['costs'])
pyplot.plot(costs)
pyplot.ylabel('cost')
pyplot.xlabel('iterations (per hundreds)')
pyplot.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
pyplot.show()

# learning_rate = 0.001, 0.0005
# num_iterations = 1000, 300

# change this to the name of your image file
my_image = "test_negativo_2.png"   

# We preprocess the image to fit your algorithm.
#DASD-818
#ONIN-050
# https://hpjav.tv/126771/jul-170
# https://hpjav.tv/109304/dasd-606
# https://hpjav.tv/118156/vrtm-475
# https://hpjav.tv/74331/rtp-024
# https://hpjav.tv/124110/gcf-007c
# https://hpjav.tv/122901/hhkl-009
num_px = 256
fname = "/home/brandonC/Documents/ML/Machine_Learning/Logistic_Regression/" + my_image
image = np.array(Image.open(fname).resize((num_px, num_px)).convert("RGB"))
pyplot.imshow(image)
pyplot.show()
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T
my_predicted_image = output_predict(logistic_regression_model["w"], logistic_regression_model["b"], image)

print("y = " + str(np.squeeze(my_predicted_image)) )

