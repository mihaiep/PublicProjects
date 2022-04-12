from Nets.MTCNN.CustomMTCNN import cMTCNN
from Nets.SCNN.scnn import SCNN
from Utils.utils import GPU_run, CPU_run
import tensorflow as tf
import numpy as np


def splitList(inputList: list, outLists=2):
    """Split inputList in [outLists] lists.\n
    Return an array of arrays"""
    count = 0
    outputList = []
    for i in range(0, outLists):
        outputList.append([])
    for i in inputList:
        outputList[count % outLists].append(i)
        count = count + 1
    return outputList


def GenerateDataForTrainSCNN(net: cMTCNN = None, files: str = None, filesDir: str = "Datasets/CelebA", filesdoneDir: str = "Datasets/Train", threadIdx: str = "Main"):
    """Checks every image for faces, if in image is only 1 faces -> OK: move the face to [filesdoneDir]\n
     else eliminate it from dataset."""
    if (net is None):
        print("Net not initialized.")
        exit(-1)
    from os import rename
    from cv2 import imwrite
    count = 0
    filesNum = len(files)
    for i in files:
        count += 1
        filePath = filesDir + "/" + i
        faces, _ = net.getAlignedFaces(filePath)
        if (len(faces) == 1):
            imwrite(filePath, faces[0])
            print("[{0} - {1:.4f}% - {2}/{3}] File \"{4}\" has 1 face.".format(threadIdx, count / filesNum * 100, count, filesNum, i))
            rename(filePath, filesdoneDir + "/" + i)
        else:
            print("[{0} - {1:.4f}% - {2}/{3}] File \"{4}\" has {5} face(s).".format(threadIdx, count / filesNum * 100, count, filesNum, i, len(faces)))


def launcher_GenerateDataForTrainSCNN(launchType: str, filesDir: str = "Datasets/CelebA", filesdoneDir: str = "Datasets/Train", threads: int = None):
    """@launchType = CPU / GPU
    \nIf @launchType == CPU -> need also @threads param
    \nMain function that coordinates the launch of MTCNN network over initial dataset."""
    import os
    import pathlib
    if not pathlib.Path(filesdoneDir).absolute().exists():
        os.mkdir(filesdoneDir)
    imagesList = os.listdir(filesDir)
    print("Launch type: {0}".format(launchType))
    if launchType == "GPU":
        GPU_run()
        mtcnn_net = cMTCNN()
        GenerateDataForTrainSCNN(net=mtcnn_net, files=imagesList, filesDir=filesDir, filesdoneDir=filesdoneDir, threadIdx="GPU")
    elif launchType == "CPU":
        if threads is None:
            print("Threads number not initialized.")
            return None
        import threading
        CPU_run()
        if threads == 1 or threads == 0:
            mtcnn_net = cMTCNN()
            GenerateDataForTrainSCNN(net=mtcnn_net, files=imagesList, filesDir=filesDir, filesdoneDir=filesdoneDir)
        else:
            lists = splitList(imagesList, threads)
            mthreads = []
            nets = []
            for t in range(0, threads):
                net_tmp = cMTCNN()
                nets.append(net_tmp)
                tmp = threading.Thread(target=GenerateDataForTrainSCNN, args=(net_tmp, lists[t], filesDir, filesdoneDir, "ThreadId:" + str(t + 1)))
                mthreads.append(tmp)
                tmp.start()
            for t in mthreads:
                t.join()
    else:
        print("Unknown launch type")
        return None


class TDataGen(tf.data.Dataset):
    def __new__(cls, datasetFilePath, num_samples=-1, resize=None, dataType=tf.dtypes.uint8, outputSize=2):
        '''@resize\n
        - None: resize=SCNN.input_shape
        - (-1,-1): no resize
        - (X,Y): resize(X,Y)'''
        if (resize is None):
            resize = SCNN.input_shape
        if resize is not None:
            resize = (resize[1], resize[0])
        if outputSize == 2:
            if (resize == (-1, -1)):
                return tf.data.Dataset.from_generator(cls._generator, output_shapes=([None, None, None], [None, None, None], [], []),
                                                      output_types=(dataType, dataType, tf.dtypes.uint8, tf.dtypes.uint8),
                                                      args=(datasetFilePath, num_samples))
            return tf.data.Dataset.from_generator(cls._generator, output_shapes=([None, None, None], [None, None, None], [], []), output_types=(dataType, dataType, tf.dtypes.uint8, tf.dtypes.uint8),
                                                  args=(datasetFilePath, num_samples, outputSize, resize))
        else:
            if (resize == (-1, -1)):
                return tf.data.Dataset.from_generator(cls._generator, output_shapes=([None, None, None], [None, None, None], []), output_types=(dataType, dataType, tf.dtypes.uint8),
                                                      args=(datasetFilePath, num_samples))
            return tf.data.Dataset.from_generator(cls._generator, output_shapes=([None, None, None], [None, None, None], []), output_types=(dataType, dataType, tf.dtypes.uint8),
                                                  args=(datasetFilePath, num_samples))

    @staticmethod
    def _generator(datasetFilePath, num_samples=3, outputSize=2, resize=None):
        """Generator used to read images"""
        import cv2
        if resize is not None:
            resize = tuple(resize)
        file = open(datasetFilePath, "r")
        count = 0
        for f in file.readlines():
            vals = f.replace("\r", "").replace("\n", "").split(",")
            img1 = cv2.imread(vals[0]).astype(np.float32)
            img2 = cv2.imread(vals[1]).astype(np.float32)
            if (resize is not None):
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                img1 = cv2.resize(img1, resize)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                img2 = cv2.resize(img2, resize)
            label1 = int(vals[2])
            label2 = int(vals[3])
            if (count == num_samples):
                break
            else:
                count += 1
            if outputSize == 2:
                yield img1, img2, label1, label2
            else:
                yield img1, img2, label1

    @staticmethod
    def remapDomain_ImageView(x, y, z):
        return ((x / 255), (y / 255)), z

    @staticmethod
    def remapDomain_ImageView_2(x, y, z, t):
        return ((x / 255), (y / 255)), [z, t]

    @staticmethod
    def remapDomain_Data_AroundZero(x, y, z):
        return ((x - 127.5) / 128, (y - 127.5) / 128), z

    @staticmethod
    def remapDomain_Data_AroundZero_2(x, y, z, t):
        return ((x - 127.5) / 128, (y - 127.5) / 128), [z, t]

    @staticmethod
    def remapDomain_DataPositiveOnly(x, y, z):
        return (x / 256, y / 256), z

    @staticmethod
    def remapDomain_DataPositiveOnly_2(x, y, z, t):
        return (x / 256, y / 256), [z, t]

    @staticmethod
    def generateFile(pathDataset: str, pathOutputFile: str = "Datasets/Files/"):
        """Generate file for training/test:
        \ndatasetPath - Path to dataset directory
        \noutputFile - Path where output file will be stored"""
        import gc
        import os
        import pathlib
        if not pathlib.Path(pathOutputFile).absolute().exists(): os.mkdir(pathOutputFile)
        if (pathDataset == "" or pathDataset is None):
            print("No dataset path provided.")
            return None

        if '\\' in pathDataset: pathDataset = "/".join(pathDataset.split('\\'))
        if '\\' in pathDataset: pathOutputFile = "/".join(pathOutputFile.split('\\'))
        if (pathOutputFile[-1] != "/"): pathOutputFile += "/"
        if (pathDataset[-1] != "/"): pathDataset += "/"
        pathOutputFile = pathOutputFile + "Samples_" + pathDataset.split("/")[-2] + ".txt"
        message = "| Generating File:" + pathOutputFile + " |"
        print("-" * len(message))
        print(message)
        print("-" * len(message))
        #
        # Get files name and split them: 1/2
        #
        print("> Reading directory and splitting files...")
        datasetSelfClass, datasetOtherClass, datasetPaths, [countSame, _] = TDataGen._readAndSplit(pathDataset)
        #
        # Build Samples - Same Class
        #
        print("> Generating samples for case: Same Class...")
        output_S = TDataGen._generateSamples_Self(countSame, datasetSelfClass, datasetPaths)
        del datasetSelfClass
        gc.collect()
        #
        # Build Samples - Different Class
        #
        print("> Generating samples for case: Other Class...")
        output_O = TDataGen._generateSamples_Other(datasetOtherClass, datasetPaths)
        del datasetOtherClass, datasetPaths
        gc.collect()
        #
        # Shuffle + Printing results
        #
        print("> Combining outputs...")
        output = TDataGen._combineResults(output_O, output_S)
        print("> Shuffle and Print")
        import random
        for i in range(0, random.randint(3, 10)):
            random.shuffle(output)
        f = open(pathOutputFile, "w")
        for line in output:
            f.write(line + "\n")
            print("\r Current line: " + line, end="")
        print("")
        f.close()
        print("> Samples generated: " + str(len(output)))
        print("> Finished.\n")

    @staticmethod
    def _readAndSplit(datasetPath):
        """Read the names of each directory and file from that path and store it in the same structure
        \nSplit the images from each class in half
        \n[className1] > {file1, file2, file3 ..}
        \n[className2] > {file1, file2, file3 ..}
        \n[className3] > {file1, file2, file3 ..}
        \n..."""
        import os, random
        countSame = 0
        countOther = 0
        datasetSelfClass = {}
        datasetOtherClass = {}
        datasetPaths = {}
        # Get each class directory
        for className in os.listdir(datasetPath):
            classPath = datasetPath
            if (classPath[-1] != "/"):
                classPath = classPath + "/" + className
            else:
                classPath = classPath + className
            tmp1 = []
            tmp2 = []
            count = 1
            filesList = os.listdir(classPath)
            random.shuffle(filesList)
            filesNum = len(filesList)
            # Get images from each class
            for filename in filesList:
                # Split the images in 2 subsets corresponding to final classes: SameClass, DifferentClass
                if (count <= (filesNum + 2) // 4 * 2):
                    tmp1.append(filename)
                    count += 1
                    countSame += 1
                else:
                    tmp2.append(filename)
                    countOther += 1
            datasetSelfClass[className] = tmp1
            if (len(tmp2) != 0):
                datasetOtherClass[className] = tmp2
            datasetPaths[className] = classPath
        return datasetSelfClass, datasetOtherClass, datasetPaths, [countSame, countOther]

    @staticmethod
    def _generateSamples_Self(imagesNum, datasetSelfClass, datasetPaths):
        """Label SELF: (0,1)"""
        output = []
        done = 0
        for classId in list(datasetSelfClass):
            classList = datasetSelfClass[classId]
            outSamplesNum = len(classList) // 2
            for i in range(0, outSamplesNum):
                pair = "{0}/{1},{0}/{2},0,1".format(datasetPaths[classId], classList[i * 2], classList[i * 2 + 1])
                output.append(pair)
                done += 2
                print("\rProgress: {0:.2f}%".format(100 * done / imagesNum), end="")
        print("")
        return output

    @staticmethod
    def _getPair(maxIndex):
        """Generate 2 random numbers in between 0 and number of elements or the array: maxIndex=len(tmp)"""
        import random
        i = random.randint(0, maxIndex - 1)
        j = random.randint(0, maxIndex - 1)
        while (i == j):
            j = random.randint(0, maxIndex - 1)
        return i, j

    @staticmethod
    def _generateSamples_Other(datasetOtherClass, datasetPaths):
        """Label SELF: (1,0)"""
        output = []
        import random
        pairsUsed = []
        notUsed = []
        parseCounter = 1

        while (len(datasetOtherClass) != 0):
            print("Parsing: {0}".format(parseCounter))
            tmp = []
            # Extract 1st element of each dictionary`s element
            # If dictionary`s element as only 1 image -> remove the element
            # Else extract 1st element of the list
            for i in list(datasetOtherClass):
                if (len(datasetOtherClass[i]) == 1):
                    tmp.append(datasetOtherClass.pop(i)[0])
                else:
                    tmp.append(datasetOtherClass[i].pop(0))
            random.shuffle(tmp)
            # If the extracted list has an odd number of elements store the last one for later
            if (len(tmp) % 2 == 1):
                notUsed.append(tmp.pop(len(tmp) - 1))

            lenTmp = len(tmp)
            done = 0
            # Select 2 random images from 2 different classes - classes are not be used previously in the same grouping
            while len(tmp):
                i, j = TDataGen._getPair(len(tmp))
                checkedAllAndGood = False
                while not checkedAllAndGood:
                    checkedAllAndGood = True
                    for k in pairsUsed:
                        classId1 = k.split("-")[0]
                        classId2 = k.split("-")[1]
                        if (int(classId1) == int(tmp[i].split("_")[0]) and int(classId2) == int(tmp[j].split("_")[0])) or \
                                (int(classId1) == int(tmp[j].split("_")[0]) and int(classId2) == int(tmp[i].split("_")[0])):
                            i, j = TDataGen._getPair(len(tmp))
                            checkedAllAndGood = False
                            break
                pair = "{0}/{1},{2}/{3},1,0".format(datasetPaths[tmp[i].split("_")[0]], tmp[i], datasetPaths[tmp[j].split("_")[0]], tmp[j])
                output.append(pair)
                pairsUsed.append("{0}-{1}".format(tmp[i].split("_")[0], tmp[j].split("_")[0]))
                tmp.pop(max(i, j))
                tmp.pop(min(i, j))
                done += 2
                print("\rProgress: {0:.2f}%".format(100 * done / lenTmp), end="")
            parseCounter += 1
            print("")

        # Process skipped images to minimize the number of not used images
        if (len(notUsed) > 1):
            print("> Processing skipped items:")
            random.shuffle(notUsed)
            while len(notUsed) // 2 > 1:
                fileName1 = notUsed[0]
                filePath1 = datasetPaths[fileName1.split("_")[0]]
                fileName2 = notUsed[1]
                filePath2 = datasetPaths[fileName2.split("_")[0]]
                label = "1,0"
                if (fileName1.split("_")[0] == fileName2.split("_")[0]): label = "0,1"
                pair = "{0}/{1},{2}/{3},{4}".format(filePath1, fileName1, filePath2, fileName2, label)
                output.append(pair)
                notUsed.pop(0)
                notUsed.pop(0)
            if (len(notUsed) == 1):
                print("Skipped:")
                print(notUsed)
        return output

    @staticmethod
    def _combineResults(ar1, ar2):
        """A pre-shuffle step. Alternate elements of both arrays"""
        output = []
        done = 0
        counter1 = min(len(ar1), len(ar2))
        counter2 = max(len(ar1), len(ar2)) - counter1
        for i in range(0, counter1):
            output.append(ar1[i])
            output.append(ar2[i])
            done += 1
            print("\rProgress: {0:.2f}%".format(100 * done / (counter1 + counter2)), end="")

        b = ar1
        if (len(ar1) < len(ar2)): b = ar2
        for i in range(counter1, counter1 + counter2):
            output.append(b[i])
            done += 1
            print("\rProgress: {0:.2f}%".format(100 * done / (counter1 + counter2)), end="")
        print("")
        return output
