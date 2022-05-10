import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



x = np.random.standard_normal(4000)
y = np.random.standard_normal(4000)
data = pd.DataFrame({'x': x, 'y': y})
data_sample = data.sample(256)


hlines = dict()
vlines = dict()

plt.figure()
plt.ion()
plt.show()
plt.scatter(data['x'], data['y'])
for i in range(8):
    time.sleep(0.5)
    var = np.random.choice([0, 1])
    if var == 0:
        split = np.random.uniform(data['x'].min(), data['x'].max())
        plt.axvline(x=split, label=i + 1, color=f'C{i}')
        vlines[split] = f'C{i}'
    else:
        split = np.random.uniform(data['y'].min(), data['y'].max())
        plt.axhline(y=split, label=i + 1, color=f'C{i}')
        hlines[split] = f'C{i}'
    plt.pause(0.01)

plt.pause(10)

plt.clf()
#
plt.scatter(data_sample['x'], data_sample['y'])
for line, color in hlines.items():
    plt.axhline(y=line, color=color)

for line, color in vlines.items():
    plt.axvline(x=line, color=color)

plt.pause(50)
    
    