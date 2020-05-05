# @Time : 2020-04-18 22:04 
# @Author : Ben 
# @Version：V 0.1
# @File : main.py
# @desc : 使用迁移学习微调ResNet18识别手写数字

import torch
from torch import nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.net = self.load_net()
        self.train_data = DataLoader(datasets.MNIST("datasets/", True, self.trans, download=False), batch_size=1000,
                                     shuffle=True)
        self.test_data = DataLoader(datasets.MNIST("datasets/", False, self.trans, download=False), batch_size=10000,
                                    shuffle=True)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters())
        # Decay LR by a factor of 0.1 every 7 epochs
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def load_net(self):
        model = models.resnet18(pretrained=True)
        # for param in model.parameters():
        #     param.requires_grad = False
        output_channels = model.conv1.out_channels
        kernel_size = model.conv1.kernel_size
        stride = model.conv1.stride
        padding = model.conv1.padding
        bias = model.conv1.bias
        model.conv1 = nn.Conv2d(1, output_channels, kernel_size, stride, padding, bias=bias)
        input_num = model.fc.in_features
        model.fc = nn.Linear(input_num, 10)
        return model.to(self.device)

    def train(self):
        self.net.train()
        for epoch in range(5):
            for i, (data, label) in enumerate(self.train_data):
                data, label = data.to(self.device), label.to(self.device)
                output = self.net(data)
                loss = self.loss(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                acc = pred.eq(label.view_as(pred)).sum().item() / self.train_data.batch_size
                print(f"epoch:{epoch},i:{i},loss:{loss.item()}, acc:{acc * 100}%")

    def evaluate(self):
        self.net.eval()
        for data, label in self.test_data:
            data, label = data.to(self.device), label.to(self.device)
            output = self.net(data)

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            acc = pred.eq(label.view_as(pred)).sum().item() / self.train_data.batch_size
            print(f"evaluate acc:{acc * 100}%")


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    trainer.evaluate()
