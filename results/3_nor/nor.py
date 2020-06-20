import numpy as np
import matplotlib.pyplot as plt

n_groups = 6

fig, ax = plt.subplots()
plt.grid(axis='y', alpha = 0.5)
index = np.arange(n_groups)
bar_width = 0.5
opacity = 1

no = (0.824, 0.788, 0.824, 0.859, 0.832, 0.895)
rects1 = plt.bar(index, no, bar_width,
alpha=opacity,
color='#3366ff'
)

plt.xlabel('Training set')
plt.ylabel('Accuracy')
plt.title('Testing on NORPark')
plt.xticks(index , ('SUNNY', 'OVERCAST', 'RAINY', 'UFPR04', 'UFPR05', 'PUC'))

plt.ylim(0.6, 1)
plt.show()