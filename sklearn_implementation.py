from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# generate random 2d data
n = 4000
x = np.random.normal(10, 1, n)
y = np.random.normal(10, 1, n)

# set train fraction and anomaly fraction
train_frac = 0.75
test_start_index = int(n*train_frac)
anomaly_frac = 0.05
n_anomaly = int((1-train_frac)*anomaly_frac*n)

# create dataframe
data = pd.DataFrame({'x': x, 'y': y})

# fit anomaly classifier
iforest = IsolationForest()
iforest.fit(X = data.iloc[0:test_start_index])
class_probs = iforest.score_samples(data.iloc[test_start_index:])

# in sklearn's implementation, scores are negative.
# the more negative, the more anamolous, so we sort scores ascending
test_df = data.iloc[3000:].copy()
test_df['class_probs'] = class_probs
test_df = test_df.sort_values(by='class_probs')

# scatter 'normal' and anomalous data
plt.figure()
plt.ion()
plt.show()
plt.scatter(test_df.iloc[0:n_anomaly]['x'], test_df.iloc[0:n_anomaly]['y'], color='orange', label='anomaly')
plt.scatter(test_df.iloc[n_anomaly:]['x'], test_df.iloc[n_anomaly:]['y'], color='blue', label='normal point')
plt.legend()
plt.pause(10)







