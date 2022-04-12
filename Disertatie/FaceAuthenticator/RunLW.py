from Utils.utils import Camera, CPU_run
from Utils.EnvSettings import EnvironmentSettings
from Nets.MTCNN.CustomMTCNN import cMTCNN
from Nets.SCNN.scnn_lw import SCNN_LW
from Nets.SCNN.scnn import SCNN
import numpy as np
import cv2

##Tests
multiplier=3
EnvironmentSettings.MTCNN_shape= tuple([z * multiplier for z in EnvironmentSettings.MTCNN_shape])
EnvironmentSettings.MTCNN_minFaceSize=multiplier * EnvironmentSettings.MTCNN_minFaceSize


CPU_run()
camera = Camera()
camera.setBufferSize()
mtcnn = cMTCNN()
scnn = SCNN_LW(True)

file = open('Database/database.csv', 'r')
data = file.readlines()
dataSamples = len(data)


while (True):
    image = camera.getSample()
    image = cv2.resize(image, EnvironmentSettings.MTCNN_shape)
    faces, res = mtcnn.getAlignedFaces(image)
    if len(faces) > 0:
        print("\n")
        image=mtcnn.drawBbox(image,res[0]['box'])
        cv2.imshow("Face", faces[0])
        print(np.shape(faces[0]))
        faces[0] = SCNN.prepareImage(faces[0])
        diffPerson = 0
        results=[]
        labels=[]
        for j in data:
            res = scnn.predict(faces[0], j)
            print("{0:4.2f}% - {1}".format(res[1]*100,j.split("|")[1].replace("\n", "")))
            if np.argmax(res) == 1:
                results.append(res[1])
                labels.append(j.split("|")[1].replace("\n", ""))
                continue
            diffPerson += res[0]
        if len(results)==0:
            print("Prediction: Different Person - {0:2.2f}%".format(diffPerson / dataSamples * 100))
        else:
            idMax=np.argmax(results)
            print("Prediction: {0} - {1:2.2f}%".format(labels[idMax], results[idMax] * 100))
        print("")
        cv2.imshow("Video", image)
        cv2.waitKey(33)
    else:
        print("\r                                                                         ", end='')
        print("\rNo face detected.",end='')
        cv2.imshow("Video", image)
        cv2.waitKey(1)
