import matplotlib.pyplot as plt
import numpy as np
import statistics as st

x = np.arange(1,19,1)
y_raw = np.array([[0.4921,0.85],
[0.4821,0.8187],
[0.9523,0.6304],
[0.8817,0.7917],
[0.8851,0.7585],
[0.7604,0.8302],
[0.8145,0.6783],
[0.6481,0.7838],
[0.7859,0.7877],
[0.8315,0.7949],
[0.7366,0.8513],
[0.6611,0.8677],
[0.8581,0.7593],
[0.7562,0.8015],
[0.6732,0.8223],
[0.8006,0.7853],
[0.7423,0.7957],
[0.8021,0.6417]])
y = np.array([])
y_err = np.array([])

for i in range(0, y_raw.shape[0]):
    y_val = st.mean(y_raw[i])
    y = np.append(y, y_val)
    y_err = np.append(y_err, [y_val - min(y_raw[i]), max(y_raw[i]) - y_val])

plt.figure()
plt.grid(axis='y', alpha = 0.5)
# plt.errorbar(x, y, fmt='o')
plt.errorbar(x, y, yerr=np.reshape(y_err,(2,-1),order='F'), fmt='o',linewidth=1,
             ecolor='g', capsize=3)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(ticks=x)
plt.ylim(0.4,1)
plt.title('Train on 04 and Test on 05 (Caffe)')
plt.show()