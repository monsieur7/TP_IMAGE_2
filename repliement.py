import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def  atom(n,m,fx,fy):
    img=np.zeros((n, m)) #cree une matrice n*m vide
    x = np.array(np.arange(0,m)) # cree un vecteur d'entiers de 0 à m exclu
    y = np.arange(0,n) # cree un vecteur d'entiers de 0 à n exclu
    e1 = np.exp(1j*2*np.pi*fx*x) #  exponentielle complexe de fréquence fx
    e2 = np.exp(1j*2*np.pi*fy*y) #exponentielle complexe de fréquence fy
    for i in range(n):
        for j in range(m):
            img[i,j] = np.real(e2[i]*np.conjugate(e1[j])) #partie réelle du produit entre les deux exponentielles aux coordonnées i et j 
    return img

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


image = atom(128, 128, 0.15, 0.37)
plt.imshow(image, cmap="gray")
fourier2d(image, 1)
image2 = cv.resize(image, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
plt.imshow(image2, cmap="gray")

fourier2d(image2, 1)
plt.show()