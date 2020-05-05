# @Time : 2020-04-30 15:38 
# @Author : Ben 
# @Version：V 0.1
# @File : trainer.py
# @desc :训练类

import torch
from torch import nn
from maskrcnn.datasets import *
from torch.utils.data import DataLoader
from maskrcnn.nets import net

from maskrcnn.utils.engine import train_one_epoch, evaluate
from maskrcnn.utils import utils
import os


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = Datasets('data', get_transform(train=True))
    dataset_test = Datasets('data', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = net(num_classes)

    path = 'models/pretrained.pth'
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        torch.save(model.state_dict(), "models/pretrained.pth")
        # evaluate(model, data_loader_test, device=device)

    print("That's it!")

    # pick one image from the test set
    for count in range(len(dataset_test)):
        img, _ = dataset_test[count]

        # put the model in evaluation mode
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])

        # boxes = prediction[0]['boxes']
        # scores = prediction[0]['scores'].unsqueeze(dim=-1)
        # print(boxes.shape)
        # print(scores.shape)
        # boxes = torch.cat((boxes,scores),dim=-1)
        # print(boxes)

        masks = prediction[0]['masks'].cpu()
        detect = torch.zeros(*(masks.shape[2:]))
        for mask in masks:
            mask = mask[0]
            indexs = torch.nonzero(mask)
            for index in indexs:
                if mask[index[0], index[1]] * 255 > detect[index[0], index[1]]:
                    detect[index[0], index[1]] = mask[index[0], index[1]] * 255

        ori = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

        image = Image.fromarray(detect.byte().cpu().numpy())
        ori.save(r"images/{}-original.jpg".format(count))
        image.save(r"images/{}-detect.jpg".format(count))


if __name__ == '__main__':
    main()
