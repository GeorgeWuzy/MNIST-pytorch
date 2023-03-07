import torch 
import torch.nn as nn
import torchvision.datasets as datasets
from model import ConvNet, ResNetMNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def predict():

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # trained on RTX3080ti

    # Choose the network
    # Network = 'CNN'
    Network = 'ResNet'
    
    num_classes = 10

    test_dataset = datasets.MNIST(root='./', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1,  shuffle=False)

    # Choose the network
    if Network == 'CNN':
        model = ConvNet(num_classes).to(device)
        model.load_state_dict(torch.load('./CNN_20.ckpt'))
    elif Network == 'ResNet':
        model = ResNetMNIST(num_classes).to(device)
        model.load_state_dict(torch.load('./ResNet_20.ckpt'))
    else:
        print('Choose wrong network!')
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            plt.ion()
            plt.imshow(images.cpu().numpy().squeeze(),cmap='gray')
            plt.title("Prediction: {}    GT: {}".format(predicted.cpu().numpy()[0], labels.cpu().numpy()[0]))
            plt.pause(0.5)
            plt.close()
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

if __name__ == '__main__':
    predict()