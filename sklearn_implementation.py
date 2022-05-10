from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# generate random 2d data
n = 4000
train_frac = 0.75
test_start_index = int(n*train_frac)
x = np.random.normal(10, 1, n)
y = np.random.normal(10, 1, n)

anomaly_frac = 0.05
n_anomaly = int((1-train_frac)*anomaly_frac*n)

# create dataframe
data = pd.DataFrame({'x': x, 'y': y})

# fit anomaly classifier
iforest = IsolationForest()
iforest.fit(X = data.iloc[0:test_start_index])
class_probs = iforest.score_samples(data.iloc[test_start_index:])

# get the indexes of normal points and anomalies
# IsolationForest.predict() outputs an array with 1's and -1's, thus the addition and bool casting

test_df = data.iloc[3000:].copy()
test_df['class_probs'] = class_probs
test_df = test_df.sort_values(by='class_probs')

plt.figure()
plt.ion()
plt.show()
plt.scatter(test_df.iloc[0:n_anomaly]['x'], test_df.iloc[0:n_anomaly]['y'], color='orange', label='anomaly')
plt.scatter(test_df.iloc[n_anomaly:]['x'], test_df.iloc[n_anomaly:]['y'], color='blue', label='normal point')

plt.pause(10)


# show points classified as anomalies and normal points

# plt.scatter(data.iloc[test_start_index:][index_normals]['x'], data.iloc[test_start_index:][index_normals]['y'], color='blue', label='normal point')
# plt.scatter(data.iloc[test_start_index:][index_anomalies]['x'], data.iloc[test_start_index:][index_anomalies]['y'], color='orange', label='anomaly')
# plt.legend()






