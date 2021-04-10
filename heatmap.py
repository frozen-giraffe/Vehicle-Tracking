import numpy as np
import matplotlib.pyplot as plt

class Heatmap():

    def __init__(self, height, width):
        self.heatmap = np.zeros((height, width))

    def get(self):
        return self.heatmap
    
    def generate(self, windows):
        for i in range(len(windows)):
            y = windows[i, 0]
            x = windows[i, 1]
            size = windows[i, 2]
            self.heatmap[y:y+size, x:x+size] += 1
    
    def visualize(self):
        plt.imshow(self.heatmap, cmap='hot')
        plt.show()

    