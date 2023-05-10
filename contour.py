import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def fourier2d(img,fe):
    [height, width] =img.shape # hauteur / largeur de l'image

    f = np.abs(np.fft.fftshift(np.fft.fft2(img))) # module  de la fft 2d de l'image centré en 0
    n = width/2 # coordonées du centre de l'image
    m = height/2

    plt.figure() # nouvelle figure
    ax = plt.axes(projection='3d') # image en 3d 
    x = np.arange(-n/height, n/height, float(fe/height)) # vecteur contenant des valeurs de -n/height à n/height avec un pas de fe/height, fe étant la fréquence d'échantillonage
    y = np.arange(-m/width,m/width, float(fe/width)) # vecteur contenant des valeurs de -m/width à m/width avec un pas de fe/width 
    X, Y = np.meshgrid(x, y) # grille contenant les deux vecteurs ci desus
    print(X.shape) # affiche les dimensions de X
    ax.plot_surface(X, Y, np.sqrt(f)) # "plot" ou trace la racine carré du module de la tf en 3d
    plt.title({"Spectre - 1"}) #titre et légendes
    plt.xlabel("Fx")
    plt.ylabel("Fy")

    plt.figure() # nouvelle figure
    plt.imshow(np.log(5*f+1),extent=[-n/height, n/height, -m/width,m/width]) # affiche l'image constitué par la tf 2d de l'image multiplié par 5, auquel on ajoute 1 et on y fait le logarithme, on l'étend avec les mêmes dimensions que la première image 
    plt.colorbar() # échelle de la couleur
    plt.xlabel("Fx") # titre et légendes
    plt.ylabel("Fy")
    plt.title("Spectre - 2")
"""
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

fourier2d(rupt1, 1)
fourier2d(rupt2, 1)
fourier2d(rupt3, 1)

multipage('multipage.pdf')
"""

path = os.path.join( os.path.join(os.getcwd(), "imagesTP"),  "Metal0007GP.png")
print("path ", path)
img = cv.imread(path)
img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

plt.imshow(img, cmap="gray")
fourier2d(img, 1)
plt.show()