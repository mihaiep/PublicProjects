from Nets.SCNN.TrainingDataGenerator import launcher_GenerateDataForTrainSCNN,TDataGen

# After this method finish scripts from Dataset/Scripts should be launched
print("Primary processing with MTCNN...")
hasBeenProcessed = input("The dataset have been processed?[yes/no]:")
if (hasBeenProcessed == "no"):
    runType = input("GPU/CPU?:")
    if runType == "CPU":
        threadsNum = input("ThreadsNumber: ")
        launcher_GenerateDataForTrainSCNN(launchType=runType, threads=int(threadsNum))
    else:
        launcher_GenerateDataForTrainSCNN(launchType=runType)

# Create a directory for each class and move the images to corresponding directory according to: identity_CelebA.txt
# ./Datasets/Scripts/0_sortImages.sh

# Remove classes with less than X images
# ./Datasets/Scripts/1_removeClassesUnderNumber.sh

# Reassign a class id for the remaining directories
# ./Datasets/Scripts/2_reorderFolders.sh

# Split dataset between train and test
# ./Datasets/Scripts/3_sliptDataset.sh

# Add class id to each file name
# ./Datasets/Scripts/4_addClassToFilename.sh

# Count iamges for each class
# ./Datasets/Scripts/5_countImagesForEachClass.sh

print("Generating files for train SCNN...")
hasBeenProcessed = input("The dataset have been processed with scripts?[yes/no]:")
if (hasBeenProcessed == "yes"):
    TDataGen.generateFile("Datasets/Train")
    TDataGen.generateFile("Datasets/Test")
