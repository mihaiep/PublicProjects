from Nets.SCNN.scnn import SCNN
from Nets.MTCNN.CustomMTCNN import cMTCNN
from tensorflow.keras.models import Model
from Utils.EnvSettings import EnvironmentSettings
import cv2
import os
import pathlib


multiplier=3
EnvironmentSettings.MTCNN_shape= tuple([z * multiplier for z in EnvironmentSettings.MTCNN_shape])
EnvironmentSettings.MTCNN_minFaceSize=multiplier * EnvironmentSettings.MTCNN_minFaceSize


currentPath = str(pathlib.Path().absolute())
imagesPath = str(pathlib.Path(currentPath + "/Database/Images").absolute())
if not os.path.exists(imagesPath):
    print("Dir \"{0}\" was not found".format(imagesPath))
    exit
images = os.listdir(imagesPath)
if len(images) == 0:
    print("No images to process.")

print("\nGetting nets ready...")
mtcnn = cMTCNN()
print("MTCNN Loaded.")

scnn = SCNN(True)
outFile="Database/database.csv"
print("SCNN Loaded.")

seq = scnn.net.get_layer(index=2)
seqSize = len(seq.layers)
startLayer = seq.get_layer(index=0)
endLayer = seq.get_layer(index=(seqSize - 1))
siamese = Model(inputs=startLayer.input, outputs=endLayer.output)
print("Siamese Net Loaded.\n")

print("Creating file: Database/database.csv")
file = open(outFile, 'w')

for i in images:
    print("Processing: {0}".format(i))
    image=cv2.imread(filename=imagesPath + "/" + i)
    image = cv2.resize(image, EnvironmentSettings.MTCNN_shape)
    cv2.imshow("Img", image)
    face, _ = mtcnn.getAlignedFaces(image)
    cv2.waitKey(100)
    if len(face) > 1:
        print("Image {0} has 2 or more faces => SKIP".format(i))
        continue
    face = face[0]
    cv2.imshow("Face",face)
    cv2.waitKey(100)
    img = SCNN.prepareImage(face)
    res = siamese.predict(img)
    for idx in range(0, len(res[0])):
        if idx < len(res[0]) - 1:
            file.write("{0},".format(res[0][idx]))
        else:
            file.write("{0}|{1}\n".format(res[0][idx], i.split(".")[0]))
file.close()
