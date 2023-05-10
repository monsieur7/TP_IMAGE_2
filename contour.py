import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
N = 128

rupt1 = np.zeros((N, N,3), dtype=np.uint8)
rupt1 = cv.line(rupt1, (int(N/2), int(0)) , (int(N/2), int(N)), (255, 255, 255), 1)
rupt1 = cv.cvtColor(rupt1, cv.COLOR_RGB2GRAY)
plt.figure()
plt.title("contour vertical")
plt.imshow(rupt1, cmap="gray")


rupt2 = np.zeros((N, N,3), dtype=np.uint8)
rupt2 = cv.line(rupt2, (int(0), int(N/2)) , (int(N), int(N/2)), (255, 255, 255), 1)
rupt2 = cv.cvtColor(rupt2, cv.COLOR_RGB2GRAY)
plt.figure()
plt.title("contour horizontal")
plt.imshow(rupt2, cmap="gray")

rupt3 = np.zeros((N, N,3), dtype=np.uint8)
rupt3 = cv.line(rupt3, (int(0), int(0)) , (int(N), int(N)), (255, 255, 255), 1)
rupt3 = cv.cvtColor(rupt3, cv.COLOR_RGB2GRAY)
plt.figure()
plt.title("contour diagonal")
plt.imshow(rupt3, cmap="gray")


plt.show()