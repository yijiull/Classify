import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms, models

from util.Train import train_model
from util.Train import imshow, initialize_model

import argparse
import pickle as pk

ap = argparse.ArgumentParser()
ap.add_argument('--input', '-i', type=str, required=True, help='the path of data directory')
ap.add_argument('--model', '-m', type=str, required=True, help='{yijiull, resnet18, resnet50, densenet121}')
ap.add_argument('--num', '-n', type=int, required=True, help='the number of classes')

ap.add_argument('--save', '-s', type=str, default='./', help='the path to save the trained model')
ap.add_argument('--batch', '-b', type=int, default=64, help='batch-size')
ap.add_argument('--epochs', '-e', type=int, default=20, help='the number of epochs')

args = vars(ap.parse_args())

plt.ion()




model, input_size = initialize_model(args['model'], args['num'], False, True)

data_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor()
])
# prepare data
data_dir = args['input']
save_dir = args['save'] if args['save'].startswith('/') else os.path.join(os.getcwd(), args['save'])

img_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(img_datasets[x], batch_size=args['batch'], shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val']}
class_name = img_datasets['train'].classes
print(class_name)
with open('name.txt', 'wb+') as f:
	f.write(pk.dumps(class_name))

#get a batch of train data
#inputs, classes = next(iter(dataloaders['train']))
#out = torchvision.utils.make_grid(inputs)
#imshow(out, title=[class_name[x] for x in classes])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model = nn.DataParallel(model)  # 多个GPU


lossFunction = nn.CrossEntropyLoss() # define a Loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3) #define optimizer
schedule = optim.lr_scheduler.StepLR(optimizer, 5, 0.1) # every 5 epochs, lr is decayed by a factor of 0.1

net, loss_history, acc_history = train_model(model, lossFunction, optimizer, schedule, dataloaders, args['epochs'])
torch.save(net.state_dict(), save_dir)


#show the result
loss_history = [h for h in loss_history]
acc_history = [h for h in acc_history]
plt.subplot(121)
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Training Epochs')
plt.plot(range(1, args['epochs']), loss_history)

plt.subplot(122)
plt.title('Acc')
plt.ylabel('Acc')
plt.xlabel("Training Epochs")
plt.plot(range(1, args['Epochs']), acc_history)
plt.show()

