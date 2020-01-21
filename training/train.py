import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
import torchvision

import torch.nn.functional as F

from lossfunction.contrastive import ContrastiveLoss
from loaders.datasetBatch import SiameseNetworkDataset  # Change this import to switch between dataset loaders
#from loaders.dataseBatchAndNeg import SiameseNetworkDataset
#from loaders.datasetRandom import SiameseNetworkDataset
from models import jAlexnet
from misc.misc import Utils
from params.config import Config

import time

class Trainer:

    @staticmethod
    def train():

        print("Training process initialized...")
        print("dataset: ", Config.training_dir)

        folder_dataset = dset.ImageFolder(root=Config.training_dir)

        siamese_dataset = SiameseNetworkDataset(
            imageFolderDataset=folder_dataset,
            transform=transforms.Compose([transforms.Resize((Config.im_w, Config.im_h)),
                                                                              transforms.ToTensor()]),
            should_invert=False)

        train_dataloader = DataLoader(siamese_dataset,
                                      shuffle=False,
                                      num_workers=8, # 8
                                      batch_size=Config.train_batch_size)

        print("lr:     ", Config.lrate)
        print("batch:  ", Config.train_batch_size)
        print("epochs: ", Config.train_number_epochs)

        net = jAlexnet(Config.pretrained)  # Model dependant from import
        net.train()

        #criterion = ContrastiveLoss()

        optimizer = optim.SGD(net.parameters(), lr=Config.lrate)  # or net.net_parameters

        counter = []
        loss_history = []
        iteration_number = 0

        best_loss = 10**15  # Random big number (bigger than the initial loss)
        best_epoch = 0
        break_counter = 0  # break after 20 epochs without loss improvement


        for epoch in range(0, Config.train_number_epochs):

            average_epoch_loss = 0
            count = 0
            start_time = time.time()

            for i, data in enumerate(train_dataloader, 0):

                img0, img1, label = data
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

                #concatenated = torch.cat((data[0], data[1]), 0)
                #Utils.imshow(torchvision.utils.make_grid(concatenated))

                output1, output2, scores = net(img0, img1)

                optimizer.zero_grad()

                #loss_contrastive = criterion(output1, output2, label)
                loss_contrastive = F.binary_cross_entropy_with_logits(scores, label) # Koch last layer only

                #print(loss_contrastive)

                loss_contrastive.backward()

                optimizer.step()

                average_epoch_loss += loss_contrastive.item()
                count += 1

            end_time = time.time() - start_time
            print("eproch timer: (secs) ", end_time)

            iteration_number += 1
            average_epoch_loss = average_epoch_loss / count
            #print(count)

            print("Epoch number {}\n Current loss {}\n".format(epoch, average_epoch_loss))
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

            if average_epoch_loss < best_loss:
                best_loss = average_epoch_loss
                best_epoch = epoch
                torch.save(net, Config.best_model_path)
                print("------------------------Best epoch: ", epoch)
                break_counter = 0

            if break_counter >= 20:
                print("Training break...")
                #break

            break_counter += 1

        torch.save(net, Config.model_path)

        print("best: ", best_epoch)
        Utils.show_plot(counter, loss_history)
