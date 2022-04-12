from Nets.SCNN.scnn import SCNN
from Nets.MTCNN.CustomMTCNN import cMTCNN
from tensorflow.keras.models import Model
from Utils.EnvSettings import EnvironmentSettings
from Utils.utils import CPU_run
from Nets.SCNN.ModelsGenerator import ModelGen
import os
import pathlib

EnvironmentSettings.showSeqStruct=False
EnvironmentSettings.showOverallStruct=False
ModelGen.updateParams()
CPU_run()

multiplier=3
EnvironmentSettings.MTCNN_shape= tuple([z * multiplier for z in EnvironmentSettings.MTCNN_shape])
EnvironmentSettings.MTCNN_minFaceSize=multiplier * EnvironmentSettings.MTCNN_minFaceSize

def checkNet(path:str,faces):
    print("")
    EnvironmentSettings.SCNN_model=int(path.split("/")[-1].split("_")[0][5:])
    print("SCNN Loading Model: {0}".format(EnvironmentSettings.SCNN_model))
    scnn = SCNN(path+"/weights.h5")
    seq = scnn.net.get_layer(index=2)
    seqSize = len(seq.layers)
    startLayer = seq.get_layer(index=0)
    endLayer = seq.get_layer(index=(seqSize - 1))
    siamese = Model(inputs=startLayer.input, outputs=endLayer.output)
    print("Siamese Net Loaded.")
    file.write(path.split("/")[-1]+"\n")
    for face in faces:
        img = SCNN.prepareImage(face)
        res = siamese.predict(img)
        file.write("\t\t\tDescriptor\t")
        for idx in range(0, len(res[0])):
            if idx < len(res[0]) - 1:
                file.write("{0};".format(res[0][idx]))
            else:
                file.write("{0}\n".format(res[0][idx]))
    file.write("\tDistance\t\tDifference Square\n")
    file.write("\tDistance\t\tDifference Square\n")
    file.write("\tDistance\t\tDifference Square\n")
    file.write("\tDistance\t\tDifference Square\n")
    file.write("\tElev Acc\t{0}\n".format(path.split("/")[-1].split("_")[3].replace(",",".")))
    file.write("\tTotal Score\n")
    file.write("\n")
    file.write("\n")
    file.write("\n")
    print("Done.")





currentPath = str(pathlib.Path().absolute())
imagesPath = str(pathlib.Path(currentPath + "/Database/Images").absolute())
if not os.path.exists(imagesPath):
    print("Dir \"{0}\" was not found".format(imagesPath))
    exit
images = os.listdir(imagesPath)
if len(images) == 0:
    print("No images to process.")

faces=[]
print("\nGetting nets ready...")
mtcnn = cMTCNN()
print("MTCNN Loaded.")
print("Getting faces:")
for i in images:
    print(" - Processing: {0}".format(i))
    face, _ = mtcnn.getAlignedFaces(imagesPath + "/" + i)
    if len(face) > 1:
        print(" - Image {0} has 2 or more faces => SKIP".format(i))
        continue
    else:
        faces.append(face[0])

outFile = "Database/database_allDesc.csv"
print("Creating file: Database/database_all_test.csv")
file = open(outFile, 'w')

netsPath=str(pathlib.Path("Nets/SCNN/Data").absolute())
if not os.path.exists(netsPath):
    print("Dir \"{0}\" was not found".format(imagesPath))
    exit

netsFolders=os.listdir(netsPath)

for net in netsFolders:
    if os.path.isfile(netsPath+"/"+net): continue
    checkNet(netsPath+"/"+net,faces)

file.close()