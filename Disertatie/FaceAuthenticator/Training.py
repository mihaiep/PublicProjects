from Nets.SCNN.scnn import SCNN
from Utils.utils import GPU_run, getListFromInput, lr_generator
from Utils.EnvSettings import EnvironmentSettings
import tensorflow as tf

GPU_run()


def train_loop(at_modelNumber, at_batchsize, at_epochs, at_optimize: list = None, at_loss: list = None):
    best = {"epoch": 0, "batch:": 0, "evalacc": 0}
    epochsList = getListFromInput(at_epochs)
    batchList = getListFromInput(at_batchsize)
    modelList = getListFromInput(at_modelNumber)
    optList = at_optimize
    lossList = at_loss
    if at_optimize is None: optList = [None]
    if at_loss is None: lossList = [None]
    for model in modelList:
        for opt in optList:
            for loss in lossList:
                for epochS in epochsList:
                    for batchS in batchList:
                        EnvironmentSettings.SCNN_model = model
                        myNet = SCNN(loadModel=False)
                        evalAcc = myNet.trainNet(epochs=epochS, batchSize=batchS, cBacks=cbacks, opt=opt, loss=loss)
                        if evalAcc > best['evalacc']:
                            best['epoch'] = epochS
                            best['batch'] = batchS
                            best['evalacc'] = evalAcc
                            best['model'] = model
    print("Best Combination: ")
    print("Eval Acc:{0:.4f}% - Epochs: {1} - Batch: {2} - Model: {3}".format(best['evalacc'] * 100, best['epoch'], best['batch'], best['model']))


################################
# GlobalParams
################################l
# bkpCallback
cbacks = []
cbacks.append(tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.005, patience=10, verbose=1))
cbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_generator))

SCNN.set_LearningParams(newLr=0.00008907357, newDecayFactor=0.85, newDecayStartEpoch=8)
train_loop(at_modelNumber=[6,7,8], at_batchsize=256, at_epochs=[25], at_loss=(None))