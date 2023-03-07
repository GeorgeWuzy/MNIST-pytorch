import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import ConvNet, ResNetMNIST

def train():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # trained on RTX3080ti

    # Choose the network
    # Network = 'CNN'
    Network = 'ResNet'

    # Hyper parameters
    num_epochs = 1 # CNN —> 5:99.13%; 10:99.16%; 20:99.16%
                    # ResNet -> 5:98.85%; 10:99.17%; 20:99.17%
    num_classes = 10
    batch_size = 100
    learning_rate = 0.001

    # MNIST dataset
    train_dataset = datasets.MNIST(root='./', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./', train=False, download=True, transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,  shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,  shuffle=False)

    # Choose the network
    if Network == 'CNN':
        model = ConvNet(num_classes).to(device)
    elif Network == 'ResNet':
        model = ResNetMNIST(num_classes).to(device)
    else:
        print('Choose wrong network!')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
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
        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    if Network == 'CNN':
        torch.save(model.state_dict(), 'CNN_'+str(epoch+1)+'.ckpt')
    elif Network == 'ResNet':
        torch.save(model.state_dict(), 'ResNet_'+str(epoch+1)+'.ckpt')

if __name__ == '__main__':
    train()