import numpy as np
import matplotlib.pyplot as plt

n_groups = 6

# create plot
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
plt.title('Testing on Trondheim Parking Lot')
plt.xticks(index , ('SUNNY', 'OVERCAST', 'RAINY', 'UFPR04', 'UFPR05', 'PUC'))

# fig.legend(loc=7)
# fig.tight_layout()
# fig.subplots_adjust(right=0.75)   
plt.ylim(0.6, 1)
plt.show()
