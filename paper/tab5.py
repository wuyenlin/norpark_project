import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 4
sunny = (0, 0.97, 0.96, 0.85)
overcast= (0.92, 0, 0.95, 0.82)
rainy = (0.94, 0.97, 0, 0.92)

# create plot
fig, ax = plt.subplots()
plt.grid(axis='y', alpha = 0.5)
index = np.arange(n_groups)
bar_width = 0.25
opacity = 1

rects1 = plt.bar(index, sunny, bar_width,
alpha=opacity,
color='#3366ff',
label='Sunny')

rects2 = plt.bar(index + bar_width, overcast, bar_width,
alpha=opacity,
color='#ff6600',
label='Overcast')

rects3 = plt.bar(index + 2*bar_width, rainy, bar_width,
alpha=opacity,
color='#ffcc00',
label='Rainy')

plt.ylabel('Accuracy')
plt.title('Inter-weather expteriment (paper)')
plt.xticks(index + bar_width, ('SUNNY', 'OVERCAST', 'RAINY', 'PKLot'))
plt.legend()
plt.ylim(0.7, 1)
plt.show()