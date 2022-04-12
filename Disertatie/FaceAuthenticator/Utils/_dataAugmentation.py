import cv2
import numpy as np
import threading
import os
import pathlib


def augImage(image=None, augType: str = 'flip'):
    """
    :param image: imagePath
    :param augType: 1 = Flip 2 = Gaussian
    :return: return modified image
    """
    image = cv2.imread(image).astype(np.float32)
    if augType == 'flip':
        image = cv2.flip(image, flipCode=1)
    elif augType == "gauss":
        row, col, ch = image.shape
        mean = 0
        scale = 20
        gauss = np.random.normal(mean, scale, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        image = (image + gauss)
    elif augType == "s&p":
        row, col = image.shape[0:2]
        s_vs_p = 0.5
        amount = 0.04
        # Salt mode
        coordsX_s = []
        coordsY_s = []
        coordsX_p = []
        coordsY_p = []
        for i in range(0, int(np.ceil(amount * row * col * s_vs_p))):
            coordsX_s.append(np.random.randint(0, row - 1))
            coordsY_s.append(np.random.randint(0, col - 1))
        for i in range(0, int(np.ceil(amount * row * col * (1. - s_vs_p)))):
            coordsX_p.append(np.random.randint(0, row - 1))
            coordsY_p.append(np.random.randint(0, col - 1))
        image[coordsX_s, coordsY_s, :] = 1
        image[coordsX_p, coordsY_p, :] = 0
    elif augType == "poisson":
        image = image / 255
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        image = np.random.poisson(image * vals) / float(vals)
        image = image * 255
    elif augType == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        image = image + image * gauss
    else:
        print("Aug Type unknown.")
        exit(1)
    image[image > 255] = 255
    image[image < 0] = 0
    image = np.floor(image).astype(np.uint8)
    return image


def thread_processImages(datasetRelPath: str, procentsArray: list, methodsArray: list, startLabel: int, endLabel: int, threadId: int = 0, Debug=False, showResults=False):
    if np.sum(procentsArray) > 1:
        print("Sum of Procents array`s elements must be at less or equal with 1.")
        exit(1)
    if len(procentsArray) != len(methodsArray):
        print("Procents array and Methods array must have same length.")
        exit(1)
    if not os.path.isdir(datasetRelPath):
        print("Dataset Path: \"{0}\" does not exists.".format(datasetRelPath))
        exit(1)

    datasetPath = str(pathlib.Path(str(pathlib.Path().absolute()) + "/" + datasetRelPath).absolute())
    for label in range(startLabel, endLabel + 1):
        currentLabelPath = str(pathlib.Path(datasetPath + "/" + str(label)).absolute())
        if not os.path.isdir(currentLabelPath):
            print("[Thread: {0}] Label: {1} - Path: Not found".format(threadId, label))
            continue
        print("[Thread: {0}] Label: {1} - Path: {2}".format(threadId, label, currentLabelPath))

        dirList = os.listdir(currentLabelPath)
        np.random.shuffle(dirList)
        imagesNum = np.floor(np.array(procentsArray) * len(dirList))
        sets = len(procentsArray)
        for i in range(0, sets):
            method = methodsArray[i]
            for j in range(0, int(imagesNum[i])):
                newName = "{0}_{1}.{2}".format(dirList[0].split(".")[0],method, dirList[0].split(".")[1])
                if Debug:
                    print("[Thread: {0}] Label: {1} - Set [{2}/{3}] - ImagesInSet: {4} - ImageSrc: {5} - NewImage: {6}".format(threadId, label, i + 1, sets, int(imagesNum[i]), dirList[0], newName))
                newImage = augImage(currentLabelPath + "/" + dirList[0], method)
                if showResults:
                    cv2.imshow(dirList[0], newImage)
                    cv2.waitKey()
                    cv2.destroyWindow(dirList[0])
                cv2.imwrite(currentLabelPath + "/" + newName, newImage)
                dirList.pop(0)


datasetRelPath = '../Datasets/CelebA_Processed'
procentsArray = [1 / 4, 1 / 4]
methodsArray = ["flip", "gauss"]
totalLabels = 8117
startLabel = 1

mthreads = []
threads = 20
for t in range(0, threads):
    startLabel = startLabel
    endLabel = startLabel + totalLabels // threads
    if endLabel > totalLabels: endLabel = totalLabels
    print("{0} -> {1} - dif: {2}".format(startLabel, endLabel, endLabel - startLabel + 1))
    tmp = threading.Thread(target=thread_processImages, args=(datasetRelPath, procentsArray, methodsArray, startLabel, endLabel, t + 1, True))
    startLabel = endLabel + 1
    mthreads.append(tmp)
    tmp.start()

for t in mthreads:
    t.join()
