from Nets.SCNN.scnn_lw import SCNN_LW
from Nets.SCNN.scnn import SCNN
from Utils.EnvSettings import EnvironmentSettings
import os

print("Processing Dir:")
rootdirPath = "Nets/SCNN/Data/"
dirs=os.listdir(rootdirPath)

for dir in dirs:
    if os.path.isfile(rootdirPath+dir): continue
    dirPath=rootdirPath+dir
    print("- {0}".format(dirPath))
    files=os.listdir(dirPath)
    if not "weights.h5" in files:
        print("Dir has not weights - Skip")
        continue

    # To be modified if template name is changed
    EnvironmentSettings.SCNN_model=int(dir.split("_")[0][5:])

    scnn = SCNN(dirPath+"/weights.h5")
    print("SCNN Loaded")

    scnn_lw = SCNN_LW(False)
    print("Light SCNN Loaded.")

    print("\nCopying weights...")
    seqLayerSrc = scnn.net.get_layer(index=2)
    sizeSeqLayer = len(seqLayerSrc.layers)
    # Left
    scnn_lw.net.get_layer(index=2).set_weights(seqLayerSrc.get_weights())
    # Right does not need
    # Common
    scnn_lw.net.get_layer(index=4).set_weights(scnn.net.get_layer(index=3).get_weights())
    scnn_lw.net.get_layer(index=5).set_weights(scnn.net.get_layer(index=4).get_weights())
    print("Saving Weights...")
    scnn_lw.net.save_weights(dirPath + "/weights_LW.h5", save_format='h5')
    print("Done")
