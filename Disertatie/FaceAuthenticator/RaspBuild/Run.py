from Utils.utils import Camera, CPU_run, GPU_run
from Utils.EnvSettings import EnvironmentSettings
from Nets.MTCNN.CustomMTCNN import cMTCNN
from Nets.SCNN.scnn import SCNN
import numpy as np
import cv2
import os

GPU_run()
camera = Camera()
camera.setBufferSize()
mtcnn = cMTCNN()
scnn = SCNN(True)

imagesInDB={}
data = os.listdir("Database/Images")
dataSamples = len(data)
for i in data:
    cmpImg = cv2.imread("Database/Images/{0}".format(i))
    cmpFace, _ = mtcnn.getAlignedFaces(cmpImg)
    cmpFace = SCNN.prepareImage(cmpFace[0])
    imagesInDB[i]=cmpFace

while (True):
    image = camera.getSample()
    image = cv2.resize(image, EnvironmentSettings.MTCNN_shape)
    face, _ = mtcnn.getAlignedFaces(image)
    if len(face) > 0:
        cv2.imshow("Face", face[0])
        face = SCNN.prepareImage(face[0])
        diffPerson = 0
        notFound = True
        for j in imagesInDB:
            res = scnn.predict(face, imagesInDB[j])
            if np.argmax(res) == 1:
                print("\rPrediction: {0} - {1:2.2f}%".format(j.split(".")[0], res[1] * 100), end='')
                notFound = False
                continue
            diffPerson += res[0]
        if notFound: print("\rPrediction: Different Person - {0:2.2f}%".format(diffPerson / dataSamples * 100), end='')
    else:
        print("\rNo face detected.", end='')
    cv2.imshow("Video", image)
    cv2.resizeWindow('Video', 640 // 3, 480 // 3)
    cv2.waitKey(1)
