import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np
import sklearn.metrics as metrics

import matplotlib.pyplot as plt
from stopDS import StopDataSet

torch.manual_seed(0)

# Helper function for visualising images in our dataset
def imshow(img):
    img = img / 2 + 0.5 #unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    rgbimg = npimg[:,:,::-1]
    plt.imshow(rgbimg)
    plt.show()


#######################################################################################################################################
####     INITIALISE OUR NETWORK                                                                                                    ####
#######################################################################################################################################

class StopNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(1344, 256)
        self.fc2 = nn.Linear(256, 2)

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm2d(6)    # Batch norm after conv1
        self.bn2 = nn.BatchNorm2d(16)   # Batch norm after conv2
        # self.dropout = nn.Dropout(0.5) # Dropout before the final layer


    def forward(self, x):
        # Extract features with convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        # x = self.pool(self.relu(self.conv1(x)))
        # x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        #linear layer for classification
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
       
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # # Transformations for raw images before going to CNN
    # train_transform = transforms.Compose([transforms.ToTensor(),
    #                                 # transforms.RandomHorizontalFlip(p=0.5),
    #                                 transforms.Resize((40, 60)),
    #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                                 ])
    
    # valid_transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Resize((40, 60)),
    #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                                 ])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224)),  # Change from (40, 60) to (224, 224)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  # Change from (40, 60) to (224, 224)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    script_path = os.path.dirname(os.path.realpath(__file__))

    ###################
    ## Train dataset ##
    ###################

    train_ds = StopDataSet(os.path.join(script_path, '..', 'data', 'train_stop_signs'), '.jpg', train_transform)
    print("The train dataset contains %d images " % len(train_ds))
    print(f"The shape of an image after transform is: {train_ds[0][0].shape}")

    # Data loader nicely batches images for the training process and shuffles (if desired)
    trainloader = DataLoader(train_ds,batch_size=8,shuffle=True)
    all_y = []
    for S in trainloader:
        im, y = S    
        all_y += y.tolist()

    print(f'Input to network shape: {im.shape}')
    train_labels = [train_ds[i][1] for i in range(len(train_ds))]
    print(f"\n=== CLASS DISTRIBUTION ===")
    print(f"Train - free (0): {train_labels.count(0)}, stop (1): {train_labels.count(1)}")


    # Visualise the distribution of GT labels
    all_lbls, all_counts = np.unique(all_y, return_counts = True)
    plt.bar(all_lbls, all_counts, width = (all_lbls[1]-all_lbls[0])/2)
    plt.xlabel('Labels')
    plt.ylabel('Counts')
    plt.xticks(all_lbls)
    plt.title('Training Dataset')
    plt.show()

    # Visualise some images and print labels -- check these seem reasonable
    example_ims, example_lbls = next(iter(trainloader))
    print(' '.join(f'{example_lbls[j]}' for j in range(len(example_lbls))))
    imshow(torchvision.utils.make_grid(example_ims))

    ########################
    ## Validation dataset ##
    ########################

    val_ds = StopDataSet(os.path.join(script_path, '..', 'data', 'val_stop_signs'), '.jpg', valid_transform)
    print("The validation dataset contains %d images " % len(val_ds))

    # Data loader nicely batches images for the training process and shuffles (if desired)
    valloader = DataLoader(val_ds,batch_size=1)
    all_y = []
    for S in valloader:
        im, y = S    
        all_y += y.tolist()

    print(f'Input to network shape: {im.shape}')
    val_labels = [val_ds[i][1] for i in range(len(val_ds))]
    print(f"Val - free (0): {val_labels.count(0)}, stop (1): {val_labels.count(1)}")

    # Visualise the distribution of GT labels
    all_lbls, all_counts = np.unique(all_y, return_counts = True)
    plt.bar(all_lbls, all_counts, width = (all_lbls[1]-all_lbls[0])/2)
    plt.xlabel('Labels')
    plt.ylabel('Counts')
    plt.xticks(all_lbls)
    plt.title('Validation Dataset')
    plt.show()
    

    # stop_net = StopNet()
    stop_net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in stop_net.parameters():
        param.requires_grad = False
    stop_net.fc = nn.Linear(512, 2)
    for param in stop_net.fc.parameters():
        param.requires_grad = True

    stop_net = stop_net.to(device)

    trainable_params = sum(p.numel() for p in stop_net.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in stop_net.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    #######################################################################################################################################
    ####     INITIALISE OUR LOSS FUNCTION AND OPTIMISER                                                                                ####
    #######################################################################################################################################

    #for classification tasks
    criterion = nn.CrossEntropyLoss()
    #You could use also ADAM
    # optimizer = optim.Adam(stop_net.parameters(), lr=0.0001)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, stop_net.parameters()), lr=0.01)

    #######################################################################################################################################
    ####     TRAINING LOOP                                                                                                             ####
    #######################################################################################################################################
    losses = {'train': [], 'val': []}
    accs = {'train': [], 'val': []}
    best_acc = 0
    for epoch in range(20):  # loop over the dataset multiple times

        stop_net.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = stop_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            epoch_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch + 1} loss: {epoch_loss / len(trainloader)}')
        losses['train'] += [epoch_loss / len(trainloader)]
        accs['train'] += [100.*correct/total]
    
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in val_ds.class_labels}
        total_pred = {classname: 0 for classname in val_ds.class_labels}

        # again no gradients needed
        val_loss = 0
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = stop_net(images)
                _, predictions = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[val_ds.class_labels[label.item()]] += 1
                    total_pred[val_ds.class_labels[label.item()]] += 1

        # print accuracy for each class
        class_accs = []
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            class_accs += [accuracy]

        accs['val'] += [np.mean(class_accs)]
        losses['val'] += [val_loss/len(valloader)]

        if np.mean(class_accs) > best_acc:
            torch.save(stop_net.state_dict(), 'stop_net.pth')
            best_acc = np.mean(class_accs)

    print('Finished Training')

    plt.plot(losses['train'], label = 'Training')
    plt.plot(losses['val'], label = 'Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(accs['train'], label = 'Training')
    plt.plot(accs['val'], label = 'Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    #######################################################################################################################################
    ####     PERFORMANCE EVALUATION                                                                                                    ####
    #######################################################################################################################################
    stop_net.load_state_dict(torch.load('stop_net.pth'))
    stop_net.eval()

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = stop_net(images)
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} validation images: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in val_ds.class_labels}
    total_pred = {classname: 0 for classname in val_ds.class_labels}

    # again no gradients needed
    actual = []
    predicted = []
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = stop_net(images)
            _, predictions = torch.max(outputs, 1)

            actual += labels.tolist()
            predicted += predictions.tolist()

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[val_ds.class_labels[label.item()]] += 1
                total_pred[val_ds.class_labels[label.item()]] += 1

    cm = metrics.confusion_matrix(actual, predicted, normalize = 'true')
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=val_ds.class_labels)
    disp.plot()
    plt.show()

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f}%')

    #######################################################################################################################################
    ####     PERFORMANCE EVALUATION ON TEST SET                                                                                       ####
    #######################################################################################################################################

    # # Load test dataset ONLY when you want to evaluate
    # test_ds = StopDataSet(os.path.join(script_path, '..', 'data', 'test_stop_signs'), '.jpg', transform)
    # print("The test dataset contains %d images " % len(test_ds))
    # testloader = DataLoader(test_ds, batch_size=1)

    # # Load best model
    # stop_net.load_state_dict(torch.load('stop_net.pth'))

    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = stop_net(images)
    #         _, predicted = torch.max(outputs, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    # print(f'Accuracy of the network on the {total} test images: {100 * correct / total:.1f}%')

    # # Confusion matrix on test set
    # actual = []
    # predicted = []
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = stop_net(images)
    #         _, predictions = torch.max(outputs, 1)
    #         actual += labels.tolist()
    #         predicted += predictions.tolist()

    # cm = metrics.confusion_matrix(actual, predicted, normalize='true')
    # disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_ds.class_labels)
    # disp.plot()
    # plt.show()

if __name__=='__main__':
    main()