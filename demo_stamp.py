import torch 
from model import ConvNet, ResNetMNIST
import glob
import cv2
import numpy as np

img_size = 28
kernel_connect = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
ans = []  # 保存图片数组

def split_digits(s, prefix_name):
    s = np.rot90(s)  # 使图片逆时针旋转90°
    # show(s)
    s_copy = cv2.dilate(s, kernel_connect, iterations=1)
    s_copy2 = s_copy.copy()
    contours, hierarchy = cv2.findContours(s_copy2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 该函数可以检测出图片中物品的轮廓
    # contours：list结构，列表中每个元素代表一个边沿信息。每个元素是(x, 1, 2)的三维向量，x表示该条边沿里共有多少个像素点，第三维的那个“2”表示每个点的横、纵坐标；
    # hierarchy：返回类型是(x, 4)的二维ndarray。x和contours里的x是一样的意思。如果输入选择cv2.RETR_TREE，则以树形结构组织输出，hierarchy的四列分别对应下一个轮廓编号、上一个轮廓编号、父轮廓编号、子轮廓编号，该值为负数表示没有对应项。

    # for it in contours:
    #     print(it)
    # print("##########################")

    idx = 0
    for contour in contours:
        idx = idx + 1
        [x, y, w, h] = cv2.boundingRect(contour)  # 当得到对象轮廓后，可用boundingRect()得到包覆此轮廓的最小正矩形，
        # show(cv2.boundingRect(contour))
        digit = s_copy[y:y + h, x:x + w]
        # show(digit)
        pad_len = (h - w) // 2
        # print(pad_len)
        if pad_len > 0:
            digit = cv2.copyMakeBorder(digit, 0, 0, pad_len, pad_len, cv2.BORDER_CONSTANT,value=0)
        elif pad_len < 0:
            digit = cv2.copyMakeBorder(digit, -pad_len, -pad_len, 0, 0, cv2.BORDER_CONSTANT, value=0)

        pad = digit.shape[0] // 4  # 避免数字与边框直接相连，留出4个像素左右。
        digit = cv2.copyMakeBorder(digit, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        digit = cv2.resize(digit, (img_size, img_size), interpolation=cv2.INTER_AREA)  # 把图片缩放至28*28
        digit = np.rot90(digit, 3)  # 逆时针旋转270°将原本图片旋转为原来的水平方向
        # show(digit)
        cv2.imwrite(prefix_name + str(idx) + '.jpg', digit)
        ans.append(digit)

def predict():

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # trained on RTX3080ti

    # Choose the network
    # Network = 'CNN'
    Network = 'ResNet'
    
    num_classes = 10

    # Choose the network
    if Network == 'CNN':
        model = ConvNet(num_classes).to(device)
        model.load_state_dict(torch.load('./CNN_20.ckpt'))
    elif Network == 'ResNet':
        model = ResNetMNIST(num_classes).to(device)
        model.load_state_dict(torch.load('./ResNet_20.ckpt'))
    else:
        print('Choose wrong network!')
    
    img_list = glob.glob('./test_img/*.png')
    model.eval()
    with torch.no_grad():
        for i, image in enumerate(img_list):
            img0 = cv2.imread(image)
            img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            ret, thresh_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow('fig', thresh_img)
            # cv2.waitKey(0)
            split_digits(thresh_img, str(i+1)+"/split_")
            num_list = []
            for inp in glob.glob('./'+str(i+1)+'/*.jpg'):
                input = cv2.imread(inp)
                input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)/255.
                input = torch.Tensor(input).to(device).unsqueeze(0).unsqueeze(0)
                output = model(input)
                # print(output)
                # a=a
                _, predicted = torch.max(output.data, 1)
                predicted = predicted.cpu().numpy()[0]
                num_list.append(predicted)
            cv2.imshow(str(num_list), img0)
            cv2.waitKey(0)
            # plt.ion()
            # plt.imshow(images.cpu().numpy().squeeze(),cmap='gray')
            # plt.title("Prediction: {}".format(predicted.cpu().numpy()[0], labels.cpu().numpy()[0]))
        # print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

if __name__ == '__main__':
    predict()