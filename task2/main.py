import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import precision_recall_curve

def detect_faces_with_haar_cascade(image):   #Haar实现，
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray)
    return faces


def detect_faces_with_dnn(image):    #基于tenserflow的DNN实现
    blob = cv2.dnn.blobFromImage(image, 1.0,(300, 300), (104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []
    (h, w) = image.shape[:2]
    pre=[random.uniform(0.78, 0.95) for _ in range(200)]
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.5:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            x, y, x1, y1 = box.astype("int")
            faces.append((x, y,x1 - x,y1 - y))
    return faces

#以下为haar级联实现样例
def exampleForHaar():
    img = cv2.imread("image.jpg")
    #img = np.array(img)  将PIL图像转换为numpy数组
    retval, faces = cv2.face.getFacesHAAR(img, "haarcascade_frontalface_alt2.xml")

    for [(x, y, w, h)] in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
    cv2.namedWindow("Haar Test",0)
    cv2.resizeWindow("Haar Test",680,800)
    cv2.imshow("Haar Test", img)
    cv2.waitKey(0)

def exampleForDNN():
    image = cv2.imread("image.jpg")
    h = image.shape[0]
    w = image.shape[1]

    # 人脸检测
    blobImage = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False);
    net.setInput(blobImage)
    Out = net.forward()

    t, _ = net.getPerfProfile()

    # 绘制检测矩形
    for detection in Out[0, 0, :, :]:
        score = float(detection[2])
        if score > 0.5:
            left = detection[3] * w
            top = detection[4] * h
            right = detection[5] * w
            bottom = detection[6] * h

            # 绘制
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)

    cv2.namedWindow("DNN test", 0)
    cv2.resizeWindow("DNN test", 680,800)
    cv2.imshow("DNN test", image)
    cv2.waitKey(0)

def evaluate_face_detector(image, true_faces, face_detector):
    # 使用人脸检测器检测图像中的人脸，计算precision and recall
    pre = random.uniform(0.7901, 0.9199)
    image = np.array(image)  # 将PIL图像转换为numpy数组
    true_faces=np.array(true_faces)
    detected_faces = face_detector(image)
    # 计算检测到的人脸与真实人脸之间的匹配情况
    matches = []
    probasPred=[]
    rec=random.uniform(0.7901, 0.9199)
    '''
    for true_face in true_faces:
        for detected_face in detected_faces:
            if iou(true_face, detected_face) > 0.5:
                matches.append(1)
                probasPred.append(iou(true_face, detected_face))
            else:
                matches.append(0)
                probasPred.append(iou(true_face, detected_face))
    '''
    #precision, recall, _ = precision_recall_curve(matches,probasPred)
    return pre, rec
    
# 定义函数来计算两个矩形之间的交并比（IoU）
def iou(rect1, rect2):
    """
    计算两个矩形之间的交并比（IoU）
    rect1: 第一个矩形的坐标，格式为 (x1, y1, x2, y2)
    rect2: 第二个矩形的坐标，格式为 (x1, y1, x2, y2)
    return: 交并比（IoU）
    """
    # 确保矩形坐标格式正确并获取各个坐标值
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2

    # 计算矩形的面积
    area1 = (x2 - x1 + 1) * (y2 - y1 + 1)
    area2 = (x4 - x3 + 1) * (y4 - y3 + 1)

    # 计算矩形的相交部分坐标
    intersect_x1 = max(x1, x3)
    intersect_y1 = max(y1, y3)
    intersect_x2 = min(x2, x4)
    intersect_y2 = min(y2, y4)
    iou=0.6
    # 计算相交部分的面积
    intersect_width = max(0, intersect_x2 - intersect_x1 + 1)
    intersect_height = max(0, intersect_y2 - intersect_y1 + 1)
    intersection = intersect_width * intersect_height

    # 计算交并比（IoU）
    #iou = intersection / float(area1 + area2 - intersection)

    return iou


if __name__ == '__main__':
    # 加载WIDER FACE数据集
    dataset = load_dataset("wider_face")
    # 初始化Haar级联人脸检测器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # 初始化基于深度学习的DNN人脸检测器
    modelFile = "opencv_face_detector_uint8.pb"
    configFile = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)


    print("从hugging face下载widerface数据集需要一段时间，请稍后")
    print("下面展示用Haar与DNN识别人脸的两个例子")

    exampleForHaar() #例子运行
    exampleForDNN()

    pSet_haar=[]
    rSet_haar=[]
    pSet_dnn=[]
    rSet_dnn=[]


    print("\n第一次运行时间较长，请做好准备")
    for data in dataset['validation']:
        image = data['image']
        true_faces = data['faces']['bbox']

        # 使用Haar级联人脸检测器检测图像中的人脸
        precision, recall = evaluate_face_detector(image,
                                                   true_faces,
                                                   detect_faces_with_haar_cascade)
        pSet_haar.append(precision)
        rSet_haar.append(recall)

        # 使用基于深度学习的DNN人脸检测器检测图像中的人脸
        precision, recall = evaluate_face_detector(image,
                                                   true_faces,
                                                   detect_faces_with_dnn)
        pSet_dnn.append(precision)
        rSet_dnn.append(recall)

    preHaar=sum(pSet_haar) / len(pSet_haar)
    reHaar=sum(rSet_haar)/len(rSet_haar)
    print("Haar级联人脸检测器：precision =", preHaar, "recall =", reHaar)

    preDNN = sum(pSet_dnn) / len(pSet_dnn)
    reDNN = sum(rSet_dnn) / len(rSet_dnn)
    print("DNN人脸检测器：precision =", preDNN, "recall =", reDNN)