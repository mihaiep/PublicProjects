class EnvironmentSettings:
    SCNN_InputShape = (72, 52, 3)
    SCNN_model = 8
    SCNN_outputNeurons = 2
    MTCNN_divFactor = 14
    MTCNN_minFaceSize = 40
    MTCNN_shape = (1920 // MTCNN_divFactor, 1080 // MTCNN_divFactor) # 160 x 90
    showSeqStruct = False
    showOverallStruct = False
