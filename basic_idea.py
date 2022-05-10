import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# generate random data
x = np.random.standard_normal(400)
y = np.random.standard_normal(400)

# create dataframe
data = pd.DataFrame({'x': x, 'y': y})

# downsample to |X| = 256 as per paper
data_sample = data.sample(256)

# scatter full set of data points
plt.figure()
plt.ion()
plt.show()
plt.scatter(data_sample['x'], data_sample['y'])

# Illustrate masking and swamping by plotting random split point on top of all data
# keep the random split points to later plot same lines over the downsampled data

# split on a random variable
for i in range(8):
    time.sleep(0.5)
    var = np.random.choice([0, 1])
    if var == 0:
        split = np.random.uniform(data_sample['x'].min(), data_sample['x'].max())
        plt.axvline(x=split, label=i + 1, color=f'C{i}')
        plt.pause(0.01)
    else:
        split = np.random.uniform(data_sample['y'].min(), data_sample['y'].max())
        plt.axhline(y=split, label=i + 1, color=f'C{i}')
        plt.pause(0.01)
plt.pause(30)