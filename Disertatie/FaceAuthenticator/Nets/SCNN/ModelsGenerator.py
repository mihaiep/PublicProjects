import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Lambda, InputLayer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as kBackend
from Utils.EnvSettings import EnvironmentSettings


class ModelGen():
    modelName = ""
    showStruct = EnvironmentSettings.showSeqStruct
    showOverallStruct = EnvironmentSettings.showOverallStruct

    @staticmethod
    def updateParams():
        ModelGen.showStruct = EnvironmentSettings.showSeqStruct
        ModelGen.showOverallStruct = EnvironmentSettings.showOverallStruct

    @staticmethod
    def getModel(type, inputShape, outputNeurons=2):
        ModelGen.modelName = "Model{0}".format(type)
        switch = [ModelGen._getSeqModel0, ModelGen._getSeqModel1, ModelGen._getSeqModel2,  # Original Structures
                  ModelGen._getSeqModel3, ModelGen._getSeqModel4, ModelGen._getSeqModel5,
                  ModelGen._getSeqModel6, ModelGen._getSeqModel7, ModelGen._getSeqModel8]

        # Define Inputs
        left_input = Input(inputShape)
        right_input = Input(inputShape)

        # Create Siamese net
        seq_model, _ = switch[type](inputShape)
        seq_left = seq_model(left_input)
        seq_right = seq_model(right_input)

        # Distance measure Layer + Output Layer
        lambdaLayer = Lambda(lambda tensor: kBackend.abs(tensor[0] - tensor[1]))
        distance = lambdaLayer([seq_left, seq_right])
        prediction = Dense(outputNeurons, activation=tf.keras.activations.softmax)(distance)
        net = Model(inputs=[left_input, right_input], outputs=prediction, name="{0}".format(ModelGen.modelName))
        if ModelGen.showStruct: seq_model.summary()
        if ModelGen.showOverallStruct: net.summary()
        return net

    @staticmethod
    def getModelLW(type, inputShape, outputNeurons=2):
        switch = [ModelGen._getSeqModel0, ModelGen._getSeqModel1, ModelGen._getSeqModel2,  # Original Structures
                  ModelGen._getSeqModel3, ModelGen._getSeqModel4, ModelGen._getSeqModel5,
                  ModelGen._getSeqModel6, ModelGen._getSeqModel7, ModelGen._getSeqModel8]

        # Left side - normal
        left_input = Input(inputShape)
        seq_model, neurons = switch[type](inputShape)
        seq_left = seq_model(left_input)

        # Right side - optimized
        right_input = Input((neurons,))
        model_r = Sequential(name="Right")
        model_r.add(InputLayer(input_shape=neurons))
        seq_right = model_r(right_input)

        # Distance measure Layer + Output Layer
        lambdaLayer = Lambda(lambda tensor: kBackend.abs(tensor[0] - tensor[1]))
        distance = lambdaLayer([seq_left, seq_right])
        prediction = Dense(outputNeurons, activation=tf.keras.activations.softmax)(distance)
        net = Model(inputs=[left_input, right_input], outputs=prediction)
        if ModelGen.showStruct: seq_model.summary()
        if ModelGen.showOverallStruct: net.summary()
        return net

    @staticmethod
    def _getSeqModel0(inputShape):
        # Define siamese net structure
        model = Sequential(name=ModelGen.modelName)
        model.add(Conv2D(32, (7, 7), activation=tf.keras.activations.relu, input_shape=inputShape, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(64, (5, 5), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(128, (3, 3), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(128, (3, 3), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(256, (2, 2), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(160, activation=tf.keras.activations.tanh))
        return model, 160

    @staticmethod
    def _getSeqModel1(inputShape):
        # Define siamese net structure
        model = Sequential(name=ModelGen.modelName)
        model.add(Conv2D(20, (7, 7), activation=tf.keras.activations.relu, input_shape=inputShape, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(40, (5, 5), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(60, (3, 3), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(80, (3, 3), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(100, (2, 2), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(100, activation=tf.keras.activations.tanh))
        return model, 100

    @staticmethod
    def _getSeqModel2(inputShape):
        # Define siamese net structure
        model = Sequential(name=ModelGen.modelName)
        model.add(Conv2D(20, (3, 3), activation=tf.keras.activations.relu, input_shape=inputShape, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(40, (3, 3), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(60, (3, 3), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(80, (3, 3), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(100, (3, 3), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(100, activation=tf.keras.activations.tanh))
        return model, 100

    @staticmethod
    def _getSeqModel3(inputShape):
        # Define siamese net structure
        model = Sequential(name=ModelGen.modelName)
        model.add(Conv2D(20, (5, 5), activation=tf.keras.activations.relu, input_shape=inputShape, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(40, (5, 5), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(60, (5, 5), activation=tf.keras.activations.relu, padding='same'))
        model.add(Conv2D(60, (3, 3), activation=tf.keras.activations.relu))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(100, (5, 5), activation=tf.keras.activations.relu, padding='same'))
        model.add(Conv2D(100, (3, 3), activation=tf.keras.activations.relu))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(100, activation=tf.keras.activations.tanh))
        return model, 100

    @staticmethod
    def _getSeqModel4(inputShape):
        # Define siamese net structure
        model = Sequential(name=ModelGen.modelName)
        model.add(Conv2D(40, (3, 3), activation=tf.keras.activations.relu, input_shape=inputShape, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(40, (5, 5), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(40, (3, 3), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(40, (3, 3), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(40, (2, 2), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(160, activation=tf.keras.activations.tanh))
        return model, 160

    @staticmethod
    def _getSeqModel5(inputShape):
        # Define siamese net structure
        model = Sequential(name=ModelGen.modelName)
        model.add(Conv2D(30, (7, 7), activation=tf.keras.activations.relu, input_shape=inputShape, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(60, (5, 5), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(120, (3, 3), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(60, (3, 3), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(60, (2, 2), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(160, activation=tf.keras.activations.tanh))
        model.add(Dense(160, activation=tf.keras.activations.tanh))
        return model, 160


    @staticmethod
    def _getSeqModel6(inputShape):
        # Define siamese net structure
        model = Sequential(name=ModelGen.modelName)
        model.add(Conv2D(25, (3, 3), activation=tf.keras.activations.relu, input_shape=inputShape, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(50, (3, 3), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(100, (3, 3), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(200, (3, 3), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(300, (3, 3), activation=tf.keras.activations.relu, padding='same'))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(150, activation=tf.keras.activations.tanh))
        return model, 150

    @staticmethod
    def _getSeqModel7(inputShape):
        # Define siamese net structure
        model = Sequential(name=ModelGen.modelName)
        model.add(Conv2D(25, (3, 3), activation=tf.keras.activations.relu, input_shape=inputShape))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(50, (3, 3), activation=tf.keras.activations.relu ))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(100, (3, 3), activation=tf.keras.activations.relu))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(200, (3, 3), activation=tf.keras.activations.relu))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(150, activation=tf.keras.activations.tanh))
        return model, 150

    @staticmethod
    def _getSeqModel8(inputShape):
        # Define siamese net structure
        model = Sequential(name=ModelGen.modelName)
        model.add(Conv2D(25, (3, 3), activation=tf.keras.activations.relu, input_shape=inputShape))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(50, (3, 3), activation=tf.keras.activations.relu))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(100, (3, 3), activation=tf.keras.activations.relu,padding='same'))
        model.add(Conv2D(100, (3, 3), activation=tf.keras.activations.relu))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Conv2D(200, (3, 3), activation=tf.keras.activations.relu,padding='same'))
        model.add(Conv2D(200, (3, 3), activation=tf.keras.activations.relu))
        model.add(MaxPooling2D(strides=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(150, activation=tf.keras.activations.tanh))
        return model, 150