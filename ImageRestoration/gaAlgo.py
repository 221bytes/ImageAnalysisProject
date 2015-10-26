import cv2
import random
import numpy as np

__author__ = 'necocityhunters'

crossoverProb = 60

imgOrignal = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
imgCorrupted = cv2.imread('lena_noisy.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Pristine Lena image ', imgOrignal)
height, width = imgOrignal.shape
totalPixel = height * width
blank_image = imgOrignal
k = 105
population = []
coefAmp = (30.0 / 255.0)
coefFreq = (0.01 / 255.0)
for x in range(0, 8, 1):
    individual = ["{0:08b}".format(random.randint(0, 255)),
                  "{0:08b}".format(random.randint(0, 255)),
                  "{0:08b}".format(random.randint(0, 255)), ]

    population.append(individual)


fitness = []
i = 0
for individual in population:

    myImg = imgOrignal.copy()
    wrongPixel = 0
    noiseAmp = int(individual[0], 2) * coefAmp
    noiseFreqRow = int(individual[1], 2) * coefFreq
    noiseFreqCol = int(individual[2], 2) * coefFreq
    print "noiseAmp : %f noiseFreqRow : %f  noiseFreqCol : %f" % (noiseAmp, noiseFreqRow, noiseFreqCol)

    for row in range(0, height, 1):
        for col in range(0, width, 1):
            n0 = (2.0 * np.pi * noiseFreqRow * float(row))
            n1 = (2.0 * np.pi * noiseFreqCol * float(col))

            n2 = n0 + n1

            sinus = np.sin(np.radians(n2))

            pxValue = noiseAmp * sinus

            myImg[row, col] += pxValue
            px = int(myImg[row, col]) - int(imgCorrupted[row, col])
            if px < 0:
                px *= -1
            wrongPixel += px

    cv2.imwrite('lena_corrupted_' + str(i) + '.png', myImg)
    i += 1
    fitness.append(k - float(wrongPixel) / float(totalPixel))
    print "Fitness = %f" % (k - float(wrongPixel) / float(totalPixel))



# noiseAmp = individual[0].value * (30.0/255.0)
# noiseFreqRow = individual[1].value * (0.01/255.0)
# noiseFreqCol = individual[2].value * (0.01/255.0)
#
# print "noiseAmp : %f noiseFreqRow : %f  noiseFreqCol : %f" % (noiseAmp, noiseFreqRow, noiseFreqCol)
#
# # noiseAmp = 27.88
# # noiseFreqRow = 9.6078 * 10**-1
# # noiseFreqCol = 8.235 * 10**-1
#
# for row in range(0, height, 1):
#     for col in range(0, width, 1):
#
#         n0 = (2.0 * np.pi * noiseFreqRow * float(row))
#         n1 = (2.0 * np.pi * noiseFreqCol * float(col))
#
#         n2 = n0 + n1
#
#         sinus = np.sin(np.radians(n2))
#
#         pxValue = noiseAmp * sinus
#
#         img[row, col] += pxValue
#
# cv2.imshow('Corrupted Lena', img)
# cv2.imwrite('lena_corrupted.bmp', img)
#
# imgOrignal = cv2.imread('lena_gray.bmp', cv2.IMREAD_GRAYSCALE)
# imgCorrupted = cv2.imread('lena_corrupted.bmp', cv2.IMREAD_GRAYSCALE)
#
# height, width = img.shape
#
#
# totalPixel = height * width
# wrongPixel = 0
# diffImage = cv2.absdiff(imgOrignal, imgOrignal)
#
# for row in range(0, height, 1):
#     for col in range(0, width, 1):
#         px = int(imgOrignal[row, col]) - int(imgOrignal[row, col])
#         if px < 0:
#             px *= -1
#         wrongPixel += px
#
# print "error rate = %f percent" % (float(wrongPixel) / float(totalPixel))

cv2.waitKey(0)

cv2.destroyAllWindows()
