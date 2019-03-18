import torch
import torch.nn as nn
import copy
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2 as cv
from torchvision import transforms, models
from model.classifier import Net
#show chinese
from matplotlib.font_manager import FontProperties  
myfont = FontProperties(fname='/usr/share/fonts/windows/msyhbd.ttc')  


plt.ion()



def freeze_parameters(model, freeze):
    '''
    freeze the weights, not to update them
    :param model:
    :param freeze:
    :return:
    '''
    if freeze:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, freeze, use_pretrained=True):
    model = None
    inputs_size = 0
    if model_name == 'yijiull':
        #TODO just a test
        model = Net(num_classes)
        inputs_size = 100  #100 * 80
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=use_pretrained)
        freeze_parameters(model, freeze)
        num_futures_in = model.fc.in_features
        model.fc = nn.Linear(num_futures_in, num_classes)
        inputs_size = 224
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=use_pretrained)
        freeze_parameters(model, freeze)
        num_futures_in = model.fc.in_features
        model.fc = nn.Linear(num_futures_in, num_classes)
        inputs_size = 224
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=use_pretrained)
        freeze_parameters(model, freeze)
        num_futures_in = model.classifier.in_features
        model.classifier = nn.Linear(num_futures_in, num_classes)
        inputs_size = 224
    return model, inputs_size

def train_model(model, criterion, optimizer, schedule, dataloader, num_epochs=20, device='cuda:0'):
    '''

    :param model: to train
    :param criterion: loss function
    :param optimizer:
    :param schedule: lr
    :param dataloader: data
    :param num_epochs:
    :return: model, Acc_history, Loss_history
    '''
    #TODO:动态显示训练过程中的Loss和Acc
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.
    Acc_history = []
    Loss_history = []
    batch_size = dataloader['train'].batch_size
    batch = {x: len(dataloader[x].dataset) // batch_size for x in ['train', 'val']}
    for epoch in range(num_epochs):
        print('Epoch: {}/{}'.format(epoch + 1, num_epochs))
        print('*' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                schedule.step()  # update parameters
                model.train()
            else:
                model.eval()

            running_loss = 0.
            running_corrects = 0

            for i, (inputs, label) in enumerate(dataloader[phase]):
                inputs = inputs.to(device)
                label = label.to(device)
                optimizer.zero_grad()  # zero the gradients

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(inputs)
                    _, idx = torch.max(output, 1)  # idx is the correct classes
                    loss = criterion(output, label)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(idx == label)
                # print(torch.sum(idx == label))
                 
                if (i+1) % 20 == 0:
                    print('{}/{} {}'.format(i+1, batch[phase], phase))

            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloader[phase].dataset)
            print('epoch:{}  {} Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                Acc_history.append(epoch_acc)
                Loss_history.append(epoch_loss)
    print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model, Acc_history, Loss_history


def imshow(inp, title=None):
    print(inp.numpy().shape)
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title, fontproperties=myfont)
    plt.pause(0.001)


def visualize_model(model, dataloader, num = 4, device='cuda:0'):
    '''
    random select from val dataset to show the predict result
    :param model:
    :param dataloader:
    :param num:
    :param device:
    :return:
    '''
    since = time.time()
    state = model.training
    model.eval()
    cnt = 0
    fig = plt.figure()
    with torch.no_grad():
        for i, (inputs, label) in enumerate(iter(dataloader['val'])):
            inputs = inputs.to(device)
            label = label.to(device)
            output = model(inputs)
            _, idx = torch.max(output, 1)
            for j in range(dataloader['val'].batch_size):
                cnt += 1
                ax = plt.subplot(num // 2, 2, cnt)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(dataloader['train'].dataset.classes[idx[j]]), fontproperties=myfont)
                imshow(inputs.cpu().data[j])
                if cnt == num:
                  
                    model.train(mode=state)
                    print('time: {}'.format(time.time() - since))
                    return
        print('time: {}'.format(time.time() - since))
        model.train(mode=state)

def predict(model, img_path, data_transform = transforms.ToTensor(), device='cuda:0'):
    '''

    :param model:
    :param img_path:
    :param data_transform:
    :param device:
    :return: the id of the predict class
    '''
    since = time.time()
    img = Image.open(img_path)  #RGB, and opencv is BGR, so if you use the opencv to read img, there need to transform
    if data_transform is not None:
        img = data_transform(img)
    img = img.float()
    #torch.tensor(x) is equivalent to x.clone().detach()
    #torch.tensor(x, requires_grad=True) is equivalent to x.clone().detach().requires_grad_(True)
    #The equivalents using clone() and detach() are recommended.
    img = img.clone().detach()
    img = img.unsqueeze(0)
    img = img.to(device)
    output = model(img)
    _, idx = torch.max(output, 1)
    print('time: {}'.format(time.time() - since))
    return idx

    #Image to np
    #arr = np.array(img)
    #cv.imshow('res', cv.cvtColor(arr, cv.COLOR_RGB2BGR))  #transform
    #cv.waitKey(0)

    #np to img
    #img = Image.fromarray(arr)


def predict_bach():
    pass


if __name__ == '__main__':
    pass
