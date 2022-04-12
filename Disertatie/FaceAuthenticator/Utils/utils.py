import os
import tensorflow as tf
import cv2
from Nets.SCNN.scnn import SCNN


# Other methos
def CPU_run():
    """
    Setup TF to run on CPU
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.get_logger().setLevel('ERROR')


def GPU_run():
    """
    Setup TF to run on GPU
    """
    tf.get_logger().setLevel('ERROR')
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)


def rangeFromTuple(a: tuple):
    if len(a) == 1:
        return range(1, a[0] + 1)
    elif len(a) == 2:
        return range(a[0], a[1] + 1)
    elif len(a) == 3:
        return range(a[0], a[1] + 1, a[2])
    exit


def getListFromInput(inp):
    if type(inp) is tuple:
        return rangeFromTuple(inp)
    elif type(inp) is not list:
        return (inp,)
    return inp


def renameAndMoveOutput():
    import os
    import datetime
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.rename('output.txt', 'Nets/SCNN/Data/output_{0}.txt'.format(time))


# Callbacks
def lr_generator(epoch):
    if epoch < SCNN.decayStartEpoch:
        print("Learning rate: {0:e}".format(SCNN.initialLearningRate))
        return SCNN.initialLearningRate
    else:
        newLR = SCNN.initialLearningRate * SCNN.decayFactor ** (epoch // SCNN.decayStartEpoch)
        print("Learning rate: {0:e}".format(newLR))
        return newLR


# Classes
class Camera:
    def __init__(self, inputSelector: int = 1):
        """
        :param inputSelector: 1 - Webcam || 2 - IP Video Camera
        """
        if inputSelector == 1:
            self.dataSource = cv2.VideoCapture(0)
            self.dataSource.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.dataSource.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        elif inputSelector == 2:
            user = 'root'
            password = 'disertatie'
            ip = '192.168.0.101'
            connectionString = 'rtsp://' + user + ':' + password + '@' + ip + '/axis-media/media.amp'
            self.dataSource = cv2.VideoCapture(connectionString)

    def runSamples(self, delay: int = 1):
        """
        :param delay: delay between 2 image fetches
        """
        while (self.dataSource.isOpened()):
            _, img = self.dataSource.read()
            cv2.imshow('Image', img)
            cv2.waitKey(delay)

    def getSample(self):
        if self.dataSource.isOpened():
            _, img = self.dataSource.read()
            return img
        return None

    def setBufferSize(self,bSize=1):
        self.dataSource.set(cv2.CAP_PROP_BUFFERSIZE, bSize)