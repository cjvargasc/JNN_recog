import torch
import torchvision
import torchvision.utils
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self, pretrained=True):

        super(SiameseNetwork, self).__init__()

        self.seq1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool2d(kernel_size=3, stride=2))

        self.seq2 = nn.Sequential(nn.Conv2d(64, 192, kernel_size=5, padding=2),
                                          nn.ReLU(inplace=True),
                                          nn.MaxPool2d(kernel_size=3, stride=2))

        self.seq3 = nn.Sequential(nn.Conv2d(192, 384, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True))

        self.seq4 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True))

        self.seq5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=3, stride=2))

        self.joint1 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True))

        self.joint2 = nn.Sequential(nn.Conv2d(192 * 2, 192, kernel_size=5, padding=2),
                                    nn.ReLU(inplace=True))

        self.joint3 = nn.Sequential(nn.Conv2d(384 * 2, 384, kernel_size=3, padding=1),
                                  nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1),
        )

    def forward(self, input1, input2):

        output1 = self.seq1(input1)
        output2 = self.seq1(input2)
        output1 = torch.cat((output1, output2), 1)
        output1 = self.joint1(output1)

        output1 = self.seq2(output1)
        output2 = self.seq2(output2)
        output1 = torch.cat((output1, output2), 1)
        output1 = self.joint2(output1)

        output1 = self.seq3(output1)
        output2 = self.seq3(output2)
        output1 = torch.cat((output1, output2), 1)
        output = self.joint3(output1)

        output = self.seq4(output)
        output = self.seq5(output)

        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.classifier(output)

        output = torch.sigmoid(output)

        return output
