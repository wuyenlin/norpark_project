import matplotlib.pyplot as plt
import numpy as np
import statistics as st

x = np.arange(1,19,1)
y_raw = np.array([[0.482,0.416,0.506],
         [0.595,0.666,0.822],
         [0.919,0.827,0.773],
         [0.865,0.650,0.835],
         [0.659,0.713,0.718],
         [0.867,0.857,0.845],
         [0.808,0.798,0.786],
         [0.717,0.836,0.812],
         [0.834,0.849,0.782],
         [0.794,0.810,0.802],
         [0.718,0.811,0.743],
         [0.766,0.784,0.819],
         [0.784,0.769,0.858],
         [0.828,0.785,0.729],
         [0.865,0.783,0.793],
         [0.855,0.815,0.822],
         [0.810,0.794,0.820],
         [0.826,0.778,0.794]])
y = np.array([])
y_err = np.array([])

for i in range(0, y_raw.shape[0]):
    y_val = st.mean(y_raw[i])
    y = np.append(y, y_val)
    y_err = np.append(y_err, [y_val - min(y_raw[i]), max(y_raw[i]) - y_val])

plt.figure()
plt.grid(axis='y', alpha = 0.5)
plt.errorbar(x, y, yerr=np.reshape(y_err,(2,-1),order='F'), fmt='o',linewidth=1,
             ecolor='g', capsize=3)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(ticks=x)
plt.ylim(0.4,1)
plt.title('Train on 05 and Test on 04')
plt.show()