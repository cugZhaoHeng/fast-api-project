import matplotlib.pyplot as plt
import numpy as np
img_data = np.random.rand(10, 10)
fig, axes = plt.subplots(1, 1, figsize=(10, 10))
axes.axis('off')
axes.imshow(img_data, cmap='gray')
plt.savefig('test2.png',  dpi=50)