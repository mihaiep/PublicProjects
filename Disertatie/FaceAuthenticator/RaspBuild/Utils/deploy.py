import os
import shutil
import pathlib

filesList = ["Run.py",
             "RunLW.py",
             "runtimeTest.py",
             "runtimeTest_LW.py",
             "Database",
             "Nets/MTCNN/",
             "Nets/SCNN/ModelsGenerator.py",
             "Nets/SCNN/scnn.py",
             "Nets/SCNN/scnn_lw.py",
             "Nets/SCNN/Data/Model.txt",
             "Nets/SCNN/Data/weights.h5",
             "Nets/SCNN/Data/weights_LW.h5",
             "Nets/SCNN/Data/NetFolder.txt",
             "Utils"]

rootDir = str(os.path.abspath("{0}/{1}".format(str(pathlib.Path().absolute()), "/..")))
deployDir = str(os.path.abspath("{0}/{1}".format(str(pathlib.Path().absolute()), "/../RaspBuild")))
if (not os.path.exists(deployDir)):
    print("Deploy dir does not exists - Creating dir: \n{0}".format(deployDir))
    os.mkdir(deployDir)
else:
    shutil.rmtree(deployDir)
    os.mkdir(deployDir)
    print("Deploy dir recreated.")

print("\nCopying files:")
for i in filesList:
    current = os.path.abspath("{0}/{1}".format(rootDir, i))
    target = os.path.abspath("{0}/{1}".format(deployDir, i))
    print("-> Current: {0}".format(current))
    splitPath = i.split("/")
    if "." in str(splitPath[-1]):
        if len(splitPath) > 1:
            tree = ""
            for j in splitPath[:-1]:
                tree = tree + j + "/"
                if not os.path.exists(deployDir + "/" + tree):
                    os.mkdir(deployDir + "/" + tree)
        shutil.copy(current, target)
    else:
        shutil.copytree(current, target)
