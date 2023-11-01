import cv2
import numpy as np

with open('Semantic Segmentation/ENET/enet-classes.txt','rt') as f:
    classNames = f.read().splitlines()

with open('Semantic Segmentation/ENET/enet-colors.txt', 'rt') as f:
    colors = f.read().splitlines()
    colors = [np.array(col.split(',')).astype('int') for col in colors]
    colors = np.array(colors, dtype='uint8')

model = cv2.dnn.readNet('Semantic Segmentation/ENET/enet-model.net')


cap = cv2.VideoCapture('Resources/street-video.mp4')
while True:
    success,img = cap.read()
    
    if success:
        blob = cv2.dnn.blobFromImage(img, 1/255, (1024, 512), 0, swapRB=True, crop=False)

        model.setInput(blob)
        output = model.forward()

        classMap = np.argmax(output[0], axis=0)

        mask = colors[classMap]
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        output = ((0.3 * img) + (0.7 * mask)).astype('uint8')

        cv2.imshow('Input',img)
        cv2.imshow('Output',output)

    if cv2.waitKey(1) & 0xff==ord('q'):
        break