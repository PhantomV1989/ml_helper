# get the image from "https://cdn.pixabay.com/photo/2017/03/27/16/50/beach-2179624_960_720.jpg"
import numpy as np
import matplotlib.pyplot as plt
import cv2

# read image in grayscale
img = cv2.imread('../data/beach-2179624_960_720.jpg', 0)

# obtain svd
U, S, V = np.linalg.svd(img)

# inspect shapes of the matrices
print(U.shape, S.shape, V.shape)

# plot images with different number of components
comps = [100, 200, 300, 500, 638]

plt.figure(figsize=(16, 8))
for i, v in enumerate(comps):
    low_rank = U[:, :v] @ np.diag(S[:v]) @ V[:v, :]
    if (i == 0):
        plt.subplot(2, 3, i + 1), plt.imshow(low_rank, cmap='gray'), plt.axis('off'), plt.title(
            "Original Image with n_components =" + str(v))
    else:
        plt.subplot(2, 3, i + 1), plt.imshow(low_rank, cmap='gray'), plt.axis('off'), plt.title(
            "n_components =" + str(v))
plt.show()