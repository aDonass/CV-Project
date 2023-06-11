import cv2
import math
import numpy as np

def DarkChannel(img, size):
    bImg,gImg,rImg = cv2.split(img)
    newImage = cv2.min(cv2.min(rImg, gImg), bImg);
    #计算每个像素位置上三个通道的最小值生成新图像
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(newImage, kernel)   #腐蚀操作
    return dark


def AtmLight(img, dark):   #atmosphere Light
    [h, w] = img.shape[:2]  #height & width
    imgSize = h * w
    num = int(max(math.floor(imgSize / 1000), 1))
    #num被选取的像素数
    darkVec = dark.ravel()
    imgVec = img.reshape(imgSize, 3)
    indices = np.argpartition(darkVec, -num)[-num:]
    atmSum = np.sum(imgVec[indices], axis=0)
    atomos = atmSum / num
    return atomos




def Guidedfilter(img,imgInput, r, eps):   #img 引导图像;imgInput输入图像
    #引导滤波,r是滤波器半径，eps是一个浮点数，表示正则化参数。
    mean_I = cv2.boxFilter(img, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(imgInput, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(img * img, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(img * imgInput, cv2.CV_64F, (r, r))
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    imgOut = mean_a * img + mean_b
    return imgOut           #对输入图像进行引导滤波



def tEstimate(img, atomos, size):
    omega = 0.95;
    img3 = img / atomos
    estiT = 1 - omega * DarkChannel(img3, size)
    # estiT,初步估计的透射率
    sc=img*255
    sc = sc.astype('uint8')
    gray = cv2.cvtColor(sc, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, estiT, r, eps)
    #引导滤波后重新估计的透射率t
    return t


def Recover(im, t, A, tx=0.1):
    t = np.maximum(t, tx)
    res = np.empty(im.shape, im.dtype)
    for i in range(3):
        res[:, :, i] = (im[:, :, i] - A[i]) / t + A[i]
    return res


if __name__ == '__main__':

    images = []
    for i in range(1, 6):
        filename = f'haze{i}.jpg'
        img = cv2.imread(filename)
        images.append(img)

    n=input("输入你想去雾的图片名(0-4),一共5张，请只输入一个数字:\n")
    n=int(n)
    if n>5 or n<0:
        n=int(input("请重新输入:"))

    sc = images[n];
    print("已读取")
    I =sc.astype('float64') / 255;

    dark = DarkChannel(I, 15);
    A = AtmLight(I, dark);
    t = tEstimate(I, A, 15);
    J = Recover(I, t, A, 0.1);

    cv2.imshow(f"dark{n}", dark);
    cv2.imshow(f"t{n}", t);
    cv2.imshow(f'I{n}', sc);
    cv2.imshow(f'J{n}', J);
    cv2.imwrite(f"./newImage/J{n}.png", J * 255);
    cv2.waitKey();
