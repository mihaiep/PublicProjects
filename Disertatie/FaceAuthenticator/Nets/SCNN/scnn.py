from Utils.EnvSettings import EnvironmentSettings
import tensorflow as tf
import numpy as np
import cv2


class SCNN:
    initialLearningRate = 0.000217
    decayFactor = 0.71
    decayStartEpoch = 4
    input_shape = EnvironmentSettings.SCNN_InputShape
    optimizer = None
    loss = None

    def __init__(self, loadModel=False):
        """
        Args:
        \n -> [loadModel] - True(default weight file)/False/Path
        """
        self.metrics = 'accuracy'
        self.net = None
        self.outputNeurons = EnvironmentSettings.SCNN_outputNeurons

        # Generating Net - Train or Run Model
        self.net = self.createNet(EnvironmentSettings.SCNN_model)

        # Loading Net Params - Train or Run Model
        if (loadModel == True):
            self.load_weights()
        elif (type(loadModel) is str):
            self.load_weights(loadModel)

    @staticmethod
    def setDefaultOptm():
        SCNN.optimizer = tf.keras.optimizers.Adam(learning_rate=SCNN.initialLearningRate)

    @staticmethod
    def setOptimizer(opt):
        SCNN.optimizer = opt(learning_rate=SCNN.initialLearningRate)

    @staticmethod
    def setDefaultLoss():
        SCNN.loss = tf.keras.losses.BinaryCrossentropy()

    @staticmethod
    def setLoss(loss):
        SCNN.loss = loss

    def createNet(self, modelNumber):
        from Nets.SCNN.ModelsGenerator import ModelGen as MG
        return MG.getModel(type=modelNumber, inputShape=self.input_shape, outputNeurons=self.outputNeurons)

    def save_net(self, trainAcc, evalAcc, epochs, trainHistory: dict = None):
        """
        Dir pattern: [ModelNum]_[OptName]_[LosName]_[AccEval]_[AccTrain]_[EpochsNum]_[Timestamp]\n
        Files in dir: \n
        -> Model
        -> Train History
        -> Weights
        """
        import datetime
        import os
        path = "Nets/SCNN/Data/"

        # Generate Directory Name
        dirName = str(self.net.name).replace("-", "") + "_"
        if type(self.optimizer) is str:
            dirName += self.optimizer.replace("_", "-") + "_"
        else:
            if self.optimizer is not None:
                dirName += self.optimizer.get_config()['name'].replace("_", "-") + "_"
        if type(self.loss) is str:
            dirName += "_" + self.loss.replace("_", "-") + "_"
        else:
            if self.loss is not None:
                dirName += self.loss.get_config()['name'].replace("_", "-") + "_"
        trainInfo = "{0:7.4f}_{1:7.4f}_{2:3d}_".format(evalAcc * 100, trainAcc * 100, epochs).replace(" ", "0").replace(".", ",")
        timestamp = datetime.datetime.now().strftime("%j-%H%M%S")
        dirName += trainInfo + timestamp
        dirPath = path + dirName
        os.mkdir(dirPath)

        print("Saving net in: " + dirPath)
        print("-> Saveing Model...")
        modelSummaryFile = open(dirPath + "/Model.txt", 'x')
        for layer in trainHistory.model.layers:
            if ("Sequential" in str(type(layer))):
                layer.summary(print_fn=lambda x: modelSummaryFile.write(x + '\n'))
                break
        modelSummaryFile.write('\n\n')
        self.net.summary(print_fn=lambda x: modelSummaryFile.write(x + '\n'))
        modelSummaryFile.close()
        print("\tDone")

        print("-> Saving History...")
        historyFile = open(dirPath + "/history.csv", 'x')
        historyFile.write('Initial_LR: {0};Decay_Factor: {1}; DecayStart_Epoch: {2}\n'.format(SCNN.initialLearningRate, SCNN.decayFactor, SCNN.decayStartEpoch))
        historyFile.write('Epoch;Loss;Accuracy;Learning_Rate' + '\n')
        for epoch in trainHistory.epoch:
            historyFile.write("{0};{1};{2};{3}".format(epoch, trainHistory.history['loss'][epoch], trainHistory.history['accuracy'][epoch], trainHistory.history['lr'][epoch]) + '\n')
        print("\tDone")
        print("-> Saving Weights...")
        self.net.save_weights(dirPath + "/weights.h5", save_format='h5')
        print("\tDone")

    def load_weights(self, path="Nets/SCNN/Data/weights.h5"):
        self.net.load_weights(path)
        print("Loaded from: " + path)

    @staticmethod
    def set_LearningParams(newLr=None, newDecayFactor=None, newDecayStartEpoch=None):
        if newLr is not None: SCNN.initialLearningRate = newLr
        if newDecayFactor is not None: SCNN.decayFactor = newDecayFactor
        if newDecayStartEpoch is not None: SCNN.decayStartEpoch = newDecayStartEpoch

    @staticmethod
    def get_LearningParams():
        return SCNN.initialLearningRate, SCNN.decayFactor, SCNN.decayStartEpoch

    @staticmethod
    def prepareImage(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SCNN.input_shape[1], SCNN.input_shape[0])).astype(np.float32)
        img = (img - 127.5) / 128
        return np.expand_dims(img, axis=0)

    def predict(self, image1, image2):
        res = self.net.predict([image1, image2])
        return res[0]

    def trainNet(self, epochs, batchSize, cBacks=None, tFileName: str = "Train", vFileName: str = "Test", trainSamples=-1, opt=None, loss=None):
        """
        :param epochs: [REQ] int
        :param batchSize: [REQ] int
        :param cBacks: Callbacks Obj
        :param tFileName: str
        :param vFileName: str
        :param trainSamples: int
        :param opt: Optimizer or Str
        :param loss: Loss or Str
        :return:
        """
        from Nets.SCNN.TrainingDataGenerator import TDataGen
        if (opt is not None):
            self.setOptimizer(opt)
        else:
            self.setDefaultOptm()
        if (loss is not None):
            self.setLoss(loss)
        else:
            self.setDefaultLoss()
        self.net.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        ################################
        # Training Paths
        ################################
        trainingFileName = tFileName
        trainingPath = "Datasets/Files/Samples_" + trainingFileName + ".txt"
        validationFileName = vFileName
        validationPath = "Datasets/Files/Samples_" + validationFileName + ".txt"

        mapFunc = TDataGen.remapDomain_Data_AroundZero
        if self.outputNeurons == 2:
            mapFunc = TDataGen.remapDomain_Data_AroundZero_2
        ################################
        # Initializing Dataset Object
        ################################
        dtObjTrain = TDataGen(datasetFilePath=trainingPath, dataType=tf.dtypes.float32, num_samples=trainSamples)
        dtObjTrain = dtObjTrain.map(mapFunc)
        dtObjTrain = dtObjTrain.batch(batch_size=batchSize)

        ################################
        # Launch Train
        ################################
        print("Starting Training...")
        if cBacks is not None:
            train_res = self.net.fit(dtObjTrain, epochs=epochs, callbacks=cBacks, shuffle=True)
        else:
            train_res = self.net.fit(dtObjTrain, epochs=epochs, shuffle=True)
        print("Training Done.")
        trainFinalAcc = train_res.history['accuracy'][-1]

        ################################
        # Initializing Dataset Object
        ################################
        dtObjEval = TDataGen(datasetFilePath=validationPath, dataType=tf.dtypes.float32)
        dtObjEval = dtObjEval.map(mapFunc)
        dtObjEval = dtObjEval.batch(batch_size=batchSize)

        ################################
        # Launch Validation
        ################################
        print("Starting Evaluation...")
        res = self.net.evaluate(dtObjEval)
        print("Evaluation done.")
        evalFinalAcc = res[1]

        if (evalFinalAcc >= 0.7):
            self.save_net(trainAcc=trainFinalAcc, evalAcc=evalFinalAcc, epochs=epochs, trainHistory=train_res)
        print('')
        print('')
        return evalFinalAcc
