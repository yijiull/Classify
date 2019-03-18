import os
import argparse
import pickle as pk

import torch
import torch.nn as nn
from util.Train import initialize_model, predict
from torchvision import transforms
from util.Train import visualize_model
from torchvision import datasets, transforms, models

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', '-m', type=str, required=True, help="path to the trained model")
    ap.add_argument('--input', '-i', type=str, help="dir or img to predicted")
    ap.add_argument('--dataset', '-d', type=str, required=True, help='dataset directory')
    args = vars(ap.parse_args())
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    assert os.path.exists(args['model']), "can't find the model"
    model, input_size = initialize_model('resnet50', 400, True, False)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args['model']))
    model.to(device)
    data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor()
    ])
    model.eval()
	# prepare data
    data_dir = args['dataset']
    img_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(img_datasets[x], batch_size=64, shuffle=True, num_workers=4)
	               for x in ['train', 'val']}
    dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val']}
    visualize_model(model, dataloaders, num=4)
    class_name = pk.loads(open('name.txt', 'rb').read())
    print(class_name)
    if args['input'] is not None:
        idx = predict(model, args['input'], data_transforms)
        print(idx, class_name[idx])
    while True:
        img = input('img_path:')
        img = img + '.jpg'
        if img[0] == 'q':
            break
        if not os.path.exists(img):
            img = os.path.join('./data/test/', img)
            if not os.path.exists(img):
                break
        idx = predict(model, img, data_transforms)
        print(idx, class_name[idx])

