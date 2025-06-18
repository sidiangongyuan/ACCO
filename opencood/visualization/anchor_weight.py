import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde


np.random.seed(0)


N, C = 100, 256


anchors = np.random.rand(N, 2)
anchors[:, 0] = anchors[:, 0] * (70 - (-70)) + (-70) 
anchors[:, 1] = anchors[:, 1] * (48 - (-48)) + (-48) 

instance_features = np.random.rand(N, C) 


weights = instance_features[:, 0] 


kde = gaussian_kde(anchors.T, weights=weights)


grid_x, grid_y = np.mgrid[-70:70:500j, -48:48:500j]
grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel()])


kde_values = kde(grid_coords).reshape(grid_x.shape)


colors = [(0, 0, 0), (1, 0, 0)] 
cmap_name = 'my_black_red'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)


plt.figure(figsize=(8, 6))
plt.imshow(kde_values, origin='lower', cmap=cm, extent=[-70, 70, -48, 48])
plt.colorbar() 
plt.title('KDE of Anchors with Foreground Weights')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.show()