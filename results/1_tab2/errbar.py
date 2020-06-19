import matplotlib.pyplot as plt
import numpy as np
import statistics as st

x = np.arange(1,19,1)
y_raw = np.array([[0.686,0.690,0.706],
         [0.768,0.775,0.780],
         [0.834,0.807,0.802],
         [0.856,0.939,0.852],
         [0.847,0.946,0.846],
         [0.929,0.916,0.931],
         [0.906,0.928,0.832],
         [0.850,0.917,0.930],
         [0.934,0.903,0.917],
         [0.874,0.918,0.837],
         [0.899,0.854,0.864],
         [0.889,0.903,0.884],
         [0.920,0.872,0.872],
         [0.885,0.916,0.892],
         [0.868,0.856,0.875],
         [0.893,0.893,0.903],
         [0.884,0.872,0.883],
         [0.900,0.885,0.880]])
y = np.array([])
y_err = np.array([])

for i in range(0, y_raw.shape[0]):
    y_val = st.mean(y_raw[i])
    y = np.append(y, y_val)
    y_err = np.append(y_err, [y_val - min(y_raw[i]), max(y_raw[i]) - y_val])
    # print(max(y_raw[i])-y_val)

# y_err = y_err.reshape(2,-1)
plt.figure()
plt.grid(axis='y', alpha = 0.5)
plt.errorbar(x, y, yerr=np.reshape(y_err,(2,-1),order='F'), fmt='o',linewidth=1,
             ecolor='g', capsize=3)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(ticks=x)
plt.title('Trained on 04 and Test on 05')
plt.show()