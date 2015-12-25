from time import sleep
import cv2
import matplotlib.pyplot as plt
import numpy as np

__author__ = 'necocityhunters'

# cap = cv2.VideoCapture('testRecord.mp4')
cap = cv2.VideoCapture(0)
a = 0
l_val = 0
s_val = 0
x_axis = []
isTrue = True
while isTrue:
    ret, frame = cap.read()

    y = frame.size

    hls_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    bgr_image = cv2.cvtColor(hls_image, cv2.COLOR_HLS2BGR)

    h, l, s = cv2.split(hls_image)

    l += l_val

    s += s_val

    average = np.sum(l) / l.size
    print(average)
    x_axis.append(average)
    hls_image[:, :, 1] = l
    hls_image[:, :, 2] = s

    key = cv2.waitKey(1)

    img = np.concatenate((bgr_image, cv2.cvtColor(hls_image, cv2.COLOR_HLS2BGR)), 0)

    cv2.imshow('test', img)

    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('i'):
        l_val += 1
    elif key & 0xFF == ord('k'):
        l_val -= 1
    if key & 0xFF == ord('l'):
        s_val += 1
    elif key & 0xFF == ord('j'):
        if s_val == 0:
            s_val = 255
        s_val -= 1


cap.release()
cv2.destroyAllWindows()

plt.plot(x_axis, label=str('light'))
plt.ylim(0, 255)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)

plt.show()
