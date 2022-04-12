from Utils.utils import Camera, CPU_run, GPU_run
from Nets.MTCNN.CustomMTCNN import cMTCNN
from Nets.SCNN.scnn_lw import SCNN_LW
from Nets.SCNN.scnn import SCNN
import tensorflow as tf
import time
import cv2
from Utils.EnvSettings import EnvironmentSettings

##Tests
multiplier=3
EnvironmentSettings.MTCNN_shape= tuple([z * multiplier for z in EnvironmentSettings.MTCNN_shape])
EnvironmentSettings.MTCNN_minFaceSize=multiplier * EnvironmentSettings.MTCNN_minFaceSize

CPU_run()
mtcnn = cMTCNN()
scnn = SCNN_LW(True)

camera=Camera()
camera.setBufferSize()
img1=None
while True:
    img1 = camera.getSample()
    img1 = cv2.resize(img1, EnvironmentSettings.MTCNN_shape)
    cv2.imshow("Video", img1)
    cv2.waitKey(1)

    faces, res = mtcnn.getAlignedFaces(img1)
    if(len(faces)==1):
        cv2.imshow("Face",faces[0])
        cv2.waitKey(1000)
        camera.dataSource.release()
        cv2.destroyWindow("Video")
        cv2.destroyWindow("Face")
        break

file = open('Database/database.csv', 'r')
data = file.readlines()
dataSamples = len(data)


mtcnn_delta_mean=0
counter_mtcnn=0
scnn_delta_mean=0
counter_scnn=0

for run in range(0,100):
    a = time.time()
    compareFace1, _ = mtcnn.getAlignedFaces(img1)
    b = time.time()
    print("MTCNN delta: "+str(b-a))
    mtcnn_delta_mean+=(b-a)
    counter_mtcnn+=1

    a = time.time()
    compareFace1=SCNN.prepareImage(compareFace1[0])
    b = time.time()
    print("Prepare Image delta: "+str(b-a))

    for i in data:
        a = time.time()
        res = scnn.predict(compareFace1, i)
        b = time.time()
        print("SCNN Delta: " + str(b - a))
        scnn_delta_mean += (b - a)
        counter_scnn+=1
    print("")

print("MTCNN Mean Runtime: {0}".format(mtcnn_delta_mean/counter_mtcnn))
print("SCNN Mean Runtime: {0}".format(scnn_delta_mean/counter_scnn))