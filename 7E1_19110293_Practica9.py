import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#Metodos para apliicar temple matching
# cv2.TM_SQDIFF (valor minimo entrega la mejor coincidencia)
# cv2.TM_SQDIFF_NORMED  (valor minimo entrega la mejor coincidencia)
# cv2.TM_CCORR  (valor maximo entrega la mejor coincidencia)
# cv2.TM_CCORR_NORMED (valor maximo entrega la mejor coincidencia)
# cv2.TM_CCOEFF (valor maximo entrega la mejor coincidencia)
# cv2.TM_CCOEFF_NORMED (valor maximo entrega la mejor coincidencia)

orig = cv2.imread('Perro.jpg') #Imagen de entrada 
imag = orig.copy()
recort = cv2.imread('Cara_perro.jpg') #Imagen del template

"---------------------------------------------------------------"

image_gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
recort_gray = cv2.cvtColor(recort, cv2.COLOR_BGR2GRAY)

res = cv2.matchTemplate(image_gray, recort_gray, cv2.TM_CCORR)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
print(min_val, max_val, min_loc, max_loc)

x1, y1 = min_loc
x2, y2 = min_loc[0] + recort.shape[1], min_loc[1] + recort.shape[0]

cv2.rectangle(imag, (x1, y1), (x2, y2), (0, 255, 0), 3)
cv2.imshow("Image", imag)
cv2.imshow("Template", recort)
cv2.waitKey(0)
cv2.destroyAllWindows()



"------------------------------------------------------------------- "


img_gris = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
recort_gris = cv2.cvtColor(recort, cv2.COLOR_BGR2GRAY)

methods = [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR,
           cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]

for method in methods:
    res = cv2.matchTemplate(img_gris,recort_gris, method = method) #Matriz resultante, necesitamos conocer el valor mas alto o bajo para obtener el mejor emparejamineto
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) #minMaxLoc no devolvera valor minimo, valor maximo y la ubicacion
    print(min_val, max_val, min_loc, max_loc)
 
    if method == cv2.TM_SQDIFF or method == cv2.TM_SQDIFF_NORMED:
        x1, y1 = min_loc # valor minimo para rodear 
        x2, y2 = min_loc[0] + recort.shape[1], min_loc[1] + recort.shape[0] #ubicacion del valor minimo
    else:
        x1, y1 = max_loc #valor maximo para rodear
        x2, y2 = max_loc[0] + recort.shape[1], max_loc[1] + recort.shape[0] #ubicacion del valor maximo

    cv2.rectangle(imag, (x1,y1), (x2,y2), (0, 255, 0), 3)
    cv2.imshow('Original', imag)
    cv2.imshow('Recortada', recort)
    imag = orig.copy()
    cv2.waitKey(0)
cv2.destroyAllWindows()



"----------------------------------------------------------------------------- "     
img = cv.imread('Perro.jpg',0)
img2 = img.copy()
template = cv.imread('Cara_perro.jpg',0)

w, h = template.shape[::-1]

methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc

    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

    plt.suptitle(meth)
    plt.show()        
           
