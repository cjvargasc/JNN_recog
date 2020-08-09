import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torchvision
#import torch.nn.functional as F
#from lossfunction.contrastive import ContrastiveLoss
from loaders.datasetBatch import SiameseNetworkDataset  # Change this import to switch between dataset loaders

from models.jAlexnet import jNetwork
from models.jAlexnet2 import jNetwork2
#from models.Alexnet import SiameseNetwork
from misc.misc import Utils
from params.config import Config

from lossfunction.contrastive import ContrastiveLoss

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

        net = jNetwork()
        #net = SiameseNetwork(pretrained=False)
        optimizer = optim.SGD(net.parameters(), lr=Config.lrate)

        counter = []
        loss_history = []
        iteration_number = 0

        best_loss = 10**15  # Random big number (bigger than the initial loss)
        starting_ep = 0
        best_epoch = 0
        break_counter = 0  # break after 20 epochs without loss improvement

        if Config.continue_training:
            print("Continue training:")
            checkpoint = torch.load(Config.model_path)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer = optim.SGD(net.parameters(), lr=Config.lrate)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            starting_ep = checkpoint['epoch'] + 1
            best_loss = checkpoint['loss']
            print("epoch: ", starting_ep, ", loss: ", best_loss)

        net.cuda()

        net.train()

        criterion = ContrastiveLoss()
        bce = nn.BCEWithLogitsLoss(reduction='mean')

        for epoch in range(starting_ep, Config.train_number_epochs):

            average_epoch_loss = 0
            count = 0
            start_time = time.time()

            for i, data in enumerate(train_dataloader, 0):

                img0, img1, label = data
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

                #concatenated = torch.cat((data[0], data[1]), 0)
                #Utils.imshow(torchvision.utils.make_grid(concatenated))

                confidence = net(img0, img1)
                #output1, output2 = net(img0, img1)

                optimizer.zero_grad()

                #loss_contrastive = criterion(output1, output2, label)
                #loss_contrastive = F.binary_cross_entropy_with_logits(confidence, label) # Koch last layer only

                loss_contrastive = bce(confidence, label)
                #loss_contrastive = criterion(confidence, label)

                #print(loss_contrastive)

                loss_contrastive.backward()

                optimizer.step()

                average_epoch_loss += loss_contrastive.item()
                count += 1

            end_time = time.time() - start_time
            print("eproch timer: (secs) ", end_time)

            average_epoch_loss = average_epoch_loss / count
            #print(count)

            print("Epoch number {}\n Current loss {}\n".format(epoch, average_epoch_loss))
            counter.append(epoch)
            loss_history.append(loss_contrastive.item())

            if epoch % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, Config.model_path + str(epoch))

            if average_epoch_loss < best_loss:
                best_loss = average_epoch_loss
                best_epoch = epoch

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, Config.best_model_path)

                print("------------------------Best epoch: ", epoch)
                break_counter = 0
            else:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, Config.model_path)

            if break_counter >= 20:
                print("Training break...")
                #break

            break_counter += 1

        print("best: ", best_epoch)
        Utils.show_plot(counter, loss_history)
