'''
Implementation of a simple neural network using TensorFlow.
Learns how to transform Celsius degrees to Farenheit degrees.
'''

import numpy as np
import tensorflow  as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

''' DATA '''
cel = np.array([-40, -10, 0, 8, 15, 22, 38], dtype = float)
far = np.array([-40, 14, 32, 46, 59, 72, 100], dtype = float)

''' Desinging the network '''
hid_lay1 = tf.keras.layers.Dense(units = 3, input_shape = [1])
hid_lay2 = tf.keras.layers.Dense(units = 3)
out_lay = tf.keras.layers.Dense(units = 1)
model = tf.keras.Sequential([hid_lay1, hid_lay2, out_lay])

''' Training '''
alpha = 0.1   # learning rate
model.compile(optimizer = tf.keras.optimizers.Adam(alpha),
              loss = 'mean_squared_error')

print('Training the network...')
training = model.fit(cel, far, epochs = 200, verbose = False)
print('The network has been trained!')

''' Plot the cost function '''
plt.plot(training.history['loss'])

''' Test it '''
print('Model prediction: 100 Celsius degrees are', model.predict([100])[0,0],
      'Farenheit degrees')
print('Actual value: 100 Celsius degrees are', 100*1.8 + 32, 'Farenheit degrees')
