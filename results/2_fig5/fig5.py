import numpy as np
import matplotlib.pyplot as plt

def plot_fig5(sunny, overcast, rainy, title):
    n_groups = 4

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

    plt.xlabel('Test set')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.xticks(index + bar_width, ('SUNNY', 'OVERCAST', 'RAINY', 'PKLot'))
    
    fig.legend(loc=7)
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)   
    plt.ylim(0.6, 1)
    plt.show()

sunny = (0, 0.949, 0.91, 0.756)
overcast= (0.877, 0, 0.904, 0.641)
rainy = (0.874, 0.942, 0, 0.709)
title = 'Inter-weather experiment (reproduction) on PKLot/test.txt'
plot_fig5(sunny, overcast, rainy, title)

sunny = (0, 0.95, 0.909, 0.786)
overcast= (0.878, 0, 0.905, 0.682)
rainy = (0.876, 0.943, 0, 0.747)
title = 'Inter-weather experiment (reproduction) on PKLot/val.txt'
plot_fig5(sunny, overcast, rainy, title)

sunny = (0, 0.97, 0.96, 0.85)
overcast= (0.92, 0, 0.95, 0.82)
rainy = (0.94, 0.97, 0, 0.92)
title = 'Inter-weather experiment (paper)'
plot_fig5(sunny, overcast, rainy, title)

sunny = (0.937, 0.95, 0.909, 0.786)
overcast= (0.916, 0.965, 0.967, 0.682)
rainy = (0.911, 0.954, 0.944, 0.747)
title = 'Inter-weather experiment (reproduction)'
plot_fig5(sunny, overcast, rainy, title)