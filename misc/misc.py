import matplotlib.pyplot as plt
import numpy as np
import torch


class Utils:

    @staticmethod
    def imshow(img, text=None, should_save=False):
        npimg = img.numpy()
        plt.axis("off")
        if text:
            plt.text(75, 8, text, style='italic', fontweight='bold',
                     bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    @staticmethod
    def show_plot(iteration, loss):
        plt.plot(iteration, loss)
        plt.show()

    @staticmethod
    def calc_mAP(cls_scores, gts, threshold):

        Utils.pad_list(cls_scores)  # Cant create a tensor with uneven row sizes
        Utils.pad_list(gts)

        # threshold = 0.5
        # cls_scores = [[0.1, 0.2, 0.3, 0.4, 0.4], [0.2, 0.3, 0.4, 0.45, 0.6]]
        # gts = [[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]]

        cls_scores = torch.FloatTensor(cls_scores)
        gts = torch.FloatTensor(gts)

        cls_scores, indices = torch.sort(cls_scores)
        gts = torch.zeros_like(gts).scatter_(dim=1, index=indices, src=gts)

        APs = torch.zeros(cls_scores.size()[0])
        precisions = torch.zeros(cls_scores.size()[0])

        cls_count = 0
        for cls in cls_scores:
            index = 1
            true_count = 0
            acum = 0
            tp = 0
            fp = 0
            for score in cls:

                if score <= threshold and gts[cls_count][index - 1] == 0:
                    true_count += 1
                    acum += true_count / index
                else:
                    acum += 0

                if (score <= threshold and gts[cls_count][index - 1] == 0):
                    tp += 1
                if (score <= threshold and gts[cls_count][index - 1] > 0):
                    fp += 1

                index += 1

            if true_count:
                APs[cls_count] = (1 / true_count) * acum
                precisions[cls_count] = tp / (tp + fp)
            else:
                APs[cls_count] = 0
                precisions[cls_count] = 0
            # print(tp, fp)
            cls_count += 1

        acum_ap = 0
        for ap in APs:
            acum_ap += ap
        mAP = acum_ap / len(APs)

        # print("APs:", APs)
        # print("Precision:", precisions)
        # print(mAP)
        return mAP.data.cpu()

    @staticmethod
    def pad_list(list):
        """ Pads a list to convert it to tensor in calc_mAP """
        max_size = 0

        # find the longest row (test class with most instances)
        for rows in list:
            size = len(rows)
            if size > max_size:
                max_size = size

        for row in list:
            while len(row) < max_size:
                row.append(66666)