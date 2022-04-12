from cv2 import imread, imshow, waitKey, line, rectangle, warpAffine, getRotationMatrix2D
from Nets.MTCNN.Original.mtcnn import MTCNN
from Utils.EnvSettings import EnvironmentSettings
import numpy as np
import math


class cMTCNN:
    def __init__(self):
        self._mtcnn = MTCNN(min_face_size=EnvironmentSettings.MTCNN_minFaceSize)

    def predict(self, image):
        """
        :param image: numpy array or path to image
        :return: results: list of dictionaries - box, confidence, keypointsleft_eye, right_eye, nose, mouth_left, mouth_right]
        """
        if (type(image) is str): image = imread(image)
        results = self._mtcnn.detect_faces(image)
        return results

    def predictAndDrawBoxes(self, image, color=(0, 155, 255)):
        """
        :param image: numpy array or path to image
        :param color: (R,G,B)
        :return: A copy of the image with bounding boxes
        """
        if (type(image) is str):
            image = imread(image)
        else:
            image = image.copy()
        results = self._mtcnn.detect_faces(image)
        for face in results:
            bounding_box = face['box']
            print("\rW:{0} H:{1}".format(bounding_box[2],bounding_box[3]),end='')
            image = self.drawBbox(image, bounding_box, color)
        return image

    def getFaces(self, image):
        """
        :param image: numpy array or path to image
        :return: faces: list of numpy arrays & results: dictionary - box, confidence, keypoints[left_eye, right_eye, nose, mouth_left, mouth_right]
        """
        if (type(image) is str): image = imread(image)
        results = self.predict(image)
        faces = []
        for face in results:
            # For each face get bounding box and
            bBox = face['box']
            face = image[bBox[1]:(bBox[1] + bBox[3]), bBox[0]:(bBox[0] + bBox[2]), :]
            faces.append(face)
        return faces, results

    def _drawImageCenterLines(self, image, color=(0, 0, 0)):
        """
        @Used for tests\n
        :param image: numpy array
        :param color: (R,G,B)
        :return: Return a copy of image with a horizontal and vertical line drawn through center of image
        """
        img = image.copy()
        H, W, _ = np.shape(image)
        center_x = W // 2
        center_y = H // 2
        img = line(img, (center_x, 0), (center_x, H), color)
        img = line(img, (0, center_y), (W, center_y), color)
        return img

    def _drawBboxCenterLines(self, image, bBox, color=(255, 0, 0)):
        """
        @Used for tests\n
        :param image: numpy array
        :param bBox: array[4]
        :param color: (R,G,B)
        :return: A copy of image with a horizontal and vertical line drawn through center of box
        """
        image = image.copy()
        H, W, _ = np.shape(image)
        bBoxCenter_x = bBox[0] + bBox[2] // 2
        bBoxCenter_y = bBox[1] + bBox[3] // 2
        image = line(image, (bBoxCenter_x, 0), (bBoxCenter_x, H), color)
        image = line(image, (0, bBoxCenter_y), (W, bBoxCenter_y), color)
        return image

    def drawBbox(self, image, bBox, color=(0, 155, 255)):
        """
        :param image: numpy array
        :param bBox: array[4] [X,Y,W,H]
        :param color: (R.G.B)
        :return: The bounding box
        """
        return rectangle(image.copy(), (bBox[0], bBox[1]), (bBox[0] + bBox[2], bBox[1] + bBox[3]), color, 2)

    def _getAlignedFace(self, image, result):
        """
        :param image: numpy array
        :param result: dictionary - box, confidence, keypoints[left_eye, right_eye, nose, mouth_left, mouth_right]
        :return: The image with the face in the center & the box
        """
        # Getting image/face info
        newBbox = result['box'].copy()
        right_eye = result['keypoints']['right_eye']
        left_eye = result['keypoints']['left_eye']
        faceCenter_x = result['box'][0] + result['box'][2] // 2
        faceCenter_y = result['box'][1] + result['box'][3] // 2
        H, W, _ = np.shape(image)

        # Translation image such that center of the face is in the center of the image
        tx = W // 2 - faceCenter_x
        ty = H // 2 - faceCenter_y
        newBbox[0] += tx
        newBbox[1] += ty
        translationMatrix = np.float32([[1, 0, tx], [0, 1, ty]])
        mImage = warpAffine(image, translationMatrix, (W, H))

        # Rotation - align eyes to be parallel with horizontal axis
        angle = math.atan(math.tan((left_eye[1] - right_eye[1]) / (left_eye[0] - right_eye[0]))) * 180 / math.pi
        rotationMatrix = getRotationMatrix2D((W // 2, H // 2), angle, 1)
        mImage = warpAffine(mImage, rotationMatrix, (W, H))
        return mImage, newBbox

    def getAlignedFaces(self, image, displayMode=False):
        """
        :param image: numpy array or path to image
        :param displayMode: False/True
        :return: Get a list with all faces aligned - faces: list of numpy arrays & results: dictionary - box, confidence, keypoints[left_eye, right_eye, nose, mouth_left, mouth_right]
        """
        if (type(image) is str): image = imread(image)
        results = self.predict(image)
        faces = []
        for result in results:
            mImage, bbox = self._getAlignedFace(image, result)
            face = mImage.copy()[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2]), :]
            if displayMode:
                # fullImg=self._drawBbox(mImage,bbox,(0,255,0))
                # fullImg = self._drawBbox(fullImg, result['box'])
                # imshow("FullImage",full)
                imshow("Face", face)
                waitKey(0)
            faces.append(face)
        return faces, results
