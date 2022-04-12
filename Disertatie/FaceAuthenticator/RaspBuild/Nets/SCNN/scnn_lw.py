from Utils.EnvSettings import EnvironmentSettings
from Nets.SCNN.ModelsGenerator import ModelGen
import numpy as np


class SCNN_LW:
    def __init__(self, loadModel):
        self.net = ModelGen.getModelLW(EnvironmentSettings.SCNN_model, EnvironmentSettings.SCNN_InputShape, EnvironmentSettings.SCNN_outputNeurons)
        if (loadModel == True):
            self.load_weights()
        elif (type(loadModel) is str):
            self.load_weights(loadModel)

    def save_net(self, dirPath):
        print("-> Saving Weights...")
        self.net.save_weights(dirPath + "/weights_LW.h5", save_format='h5')
        print("\tDone")

    def load_weights(self, path: str = "Nets/SCNN/Data/weights_LW.h5"):
        self.net.load_weights(path)
        print("Loaded from: " + path)

    def predict(self, img1, line):
        arr = np.expand_dims(np.array(line.split("|")[0].split(",")).astype(np.float), axis=0)
        res = self.net.predict([img1, arr])
        return res[0]
