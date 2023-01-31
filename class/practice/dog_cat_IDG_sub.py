import numpy as np
hong = np.load('C:/_data/hong.npy')
import pandas as pd

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = load_model('C:/_data/mcp/dogcat_0131_1305_0001-0.6932.hdf5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])



y_pred=model.predict(hong)
print(y_pred)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
if y_pred == 1:
    print('dog')
    
else:
    print('cat')
