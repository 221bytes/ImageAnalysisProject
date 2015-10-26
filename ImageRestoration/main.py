import numpy as np
import cv2
import random
import ctypes
__author__ = 'necocityhunters'

crossoverProb = 60

def set_bit(v, index, x):
    mask = 1 << index
    v &= ~mask
    if x:
        v |= mask
    return v


img = cv2.imread('lena_gray_my.bmp', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Pristine Lena image ', img)
height, width = img.shape
blank_image = img

genes = [ctypes.c_ubyte(random.randint(0, 255)),
         ctypes.c_ubyte(random.randint(0, 255)),
         ctypes.c_ubyte(random.randint(0, 255))]

noiseAmp = genes[0].value * (30.0/255.0)
noiseFreqRow = genes[1].value * (0.01/255.0)
noiseFreqCol = genes[2].value * (0.01/255.0)

print "noiseAmp : %f noiseFreqRow : %f  noiseFreqCol : %f" % (noiseAmp, noiseFreqRow, noiseFreqCol)

# noiseAmp = 27.88
# noiseFreqRow = 9.6078 * 10**-1
# noiseFreqCol = 8.235 * 10**-1

for row in range(0, height, 1):
    for col in range(0, width, 1):

        n0 = (2.0 * np.pi * noiseFreqRow * float(row))
        n1 = (2.0 * np.pi * noiseFreqCol * float(col))

        n2 = n0 + n1

        sinus = np.sin(np.radians(n2))

        pxValue = noiseAmp * sinus

        img[row, col] += pxValue

cv2.imshow('Corrupted Lena', img)
cv2.imwrite('lena_corrupted_my.bmp', img)

imgOrignal = cv2.imread('lena_gray_my.bmp', cv2.IMREAD_GRAYSCALE)
imgCorrupted = cv2.imread('lena_corrupted_my.bmp', cv2.IMREAD_GRAYSCALE)

height, width = img.shape


totalPixel = height * width
wrongPixel = 0
diffImage = cv2.absdiff(imgOrignal, imgOrignal)

for row in range(0, height, 1):
    for col in range(0, width, 1):
        px = int(imgOrignal[row, col]) - int(imgCorrupted[row, col])
        if px < 0:
            px *= -1
        wrongPixel += px

print "error rate = %f percent" % (float(wrongPixel) / float(totalPixel))

cv2.waitKey(0)

cv2.destroyAllWindows()
