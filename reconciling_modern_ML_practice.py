import mnist
from sklearn.metrics import mean_squared_error
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

configproto = tf.compat.v1.ConfigProto()
configproto.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=configproto)
tf.compat.v1.keras.backend.set_session(sess)

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

test_images_file = open('C:/Users/Bozhidar/PycharmProjects/ML4DS2-Part1/Data/MNIST/t10k-images.idx3-ubyte', 'rb')
test_labels_file = open('C:/Users/Bozhidar/PycharmProjects/ML4DS2-Part1/Data/MNIST/t10k-labels.idx1-ubyte', 'rb')
train_images_file = open('C:/Users/Bozhidar/PycharmProjects/ML4DS2-Part1/Data/MNIST/train-images.idx3-ubyte', 'rb')
train_labels_file = open('C:/Users/Bozhidar/PycharmProjects/ML4DS2-Part1/Data/MNIST/train-labels.idx1-ubyte', 'rb')

test_images = mnist.parse_idx(test_images_file)
test_labels = mnist.parse_idx(test_labels_file)
train_images = mnist.parse_idx(train_images_file)
train_labels = mnist.parse_idx(train_labels_file)

# plt.imshow(train_images[3, :, :], cmap='gray')
# plt.show()


# flatten the images to 1D vector
train_images = train_images.reshape(len(train_images), -1)
test_images = test_images.reshape(len(test_images), -1)

# take subset of MNIST
n = 4 * 10 ** 3
train_images = train_images[:n]
train_labels = train_labels[:n]
print(train_images.shape)
print(train_labels.shape)

scaler = StandardScaler()
train_images = scaler.fit_transform(train_images)
test_images = scaler.transform(test_images)

num_classes = 10
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)


def sparse_zero_one_loss(predictions, truth):
    y_pred = np.argmax(predictions, axis=1)
    y_truth = np.argmax(truth, axis=1)
    return np.sum(y_pred != y_truth) / len(y_pred)


test_scores = []
train_scores = []
number_of_parameters = []
# 1006
for h in range(10, 1000, 50):
    model = Sequential()
    model.add(Dense(h, input_dim=28 * 28))
    # model.add(Dense(1))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    model.fit(train_images, train_labels, epochs=10, verbose=2)

    predictions = model.predict(train_images, verbose=0)
    train_score = 100 * sparse_zero_one_loss(train_labels, predictions)
    train_scores.append(train_score)

    score = model.evaluate(test_images, test_labels, verbose=1)
    test_score = 100 * (1 - score[1])
    test_scores.append(test_score)

    current_number_of_parameters = (784 + 1) * h + (h + 1) * num_classes
    number_of_parameters.append(current_number_of_parameters)

    print('checkpoint:', h, current_number_of_parameters, train_score, test_score)

plt.figure()
plt.plot(number_of_parameters, test_scores, 'o-', c='b', label="Test", mfc="none", ms=8)
plt.plot(number_of_parameters, train_scores, 'o-', c='r', label="Train", mfc="none", ms=8)
plt.legend()
# plt.xticks(degrees)
plt.xlabel('Number of parameters')
plt.ylabel('RMSE')
plt.show()
