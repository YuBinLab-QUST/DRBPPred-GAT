from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#import time
#start = time.time()
import pandas as pd
import numpy as np
import sklearn.preprocessing as prep
from sklearn.preprocessing import scale,StandardScaler
import tensorflow as tf
from Autoencoder1 import Autoencoder
from DenoisingAutoencoder import MaskingNoiseAutoencoder


tf.reset_default_graph()


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

X = pd.read_csv('ALL.csv',header=None)

X1 = np.array(X)
X2 = X1[1:,1:]
X3 = scale(X2)

X_train = X3#.transpose(),header=None



n_samples,_ = np.shape(X_train)

training_epochs = 10
batch_size = 16
display_step = 1

autoencoder = Autoencoder(
    n_input = 1232,
    n_hidden1 = 256,
    n_hidden2 = 100,
    n_hidden3 = 256,
    transfer_function=tf.nn.softplus,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
#    dropout_probability = 0.95
    )

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        # Fit training using batch data
        cost = autoencoder.partial_fit(batch_xs)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%d,' % (epoch + 1),
              "Cost:", "{:.9f}".format(avg_cost))

print("Total cost: " + str(autoencoder.calc_total_cost(X_train)))
X_test_transform=autoencoder.transform(X_train)


data_csv = pd.DataFrame(data=X_test_transform)
data_csv.to_csv('ALL_Auto.csv')
#end = time.time()
#print('time:',end-start)
