import torch
from matplotlib import pyplot as plt
import os
import sys

class SemSegBoard():

    def __init__(self, checkpoint, batch_size, epoch, train_data=2975, val_data=500):
        self.model_name = checkpoint
        # self.checkpoint0 = torch.load(f"./outputs/checkpoints/{checkpoint}/{checkpoint}__e2")
        self.checkpoint = torch.load(f"./outputs/checkpoints/{checkpoint}/{checkpoint}__e{epoch}")

        self.epochs = self.checkpoint["epoch"]
        print("Epochs:", self.epochs)
        self.batch_size = batch_size
        self.train_data = train_data
        self.val_data = val_data

        self.train_loss = [loss.cpu().detach().numpy() for sublist in self.checkpoint["loss"]["train"] for loss in sublist]
        self.train_accuracy = [acc.cpu().detach().numpy() for batch_acc in self.checkpoint["accuracy"]["train"] for acc in batch_acc]
        self.train_iou_class = [iou.cpu().detach().numpy() for batch_iou in self.checkpoint["iou_class"]["train"] for iou in batch_iou]
        self.train_iou_category = [iou.cpu().detach().numpy() for batch_iou in self.checkpoint["iou_category"]["train"] for iou in batch_iou]

        # self.val_loss = [loss.cpu().detach().numpy() for sublist in self.checkpoint["loss"]["val"] for loss in sublist]
        self.val_loss = [sum([loss.cpu().detach().numpy() for loss in sublist])/len(sublist) for sublist in self.checkpoint["loss"]["val"]]
        # self.val_accuracy = [acc.cpu().detach().numpy() for batch_acc in self.checkpoint["accuracy"]["val"] for acc in batch_acc]
        self.val_accuracy = [sum([acc.cpu().detach().numpy() for acc in batch_acc])/len(batch_acc) for batch_acc in self.checkpoint["accuracy"]["val"]]
        # self.val_iou_class = [iou.cpu().detach().numpy() for batch_iou in self.checkpoint["iou_class"]["val"] for iou in batch_iou]
        self.val_iou_class = [sum([iou.cpu().detach().numpy() for iou in batch_iou])/len(batch_iou) for batch_iou in self.checkpoint["iou_class"]["val"]]
        # self.val_iou_category = [iou.cpu().detach().numpy() for batch_iou in self.checkpoint["iou_category"]["val"] for iou in batch_iou]
        self.val_iou_category = [sum([iou.cpu().detach().numpy() for iou in batch_iou])/len(batch_iou) for batch_iou in self.checkpoint["iou_category"]["val"]]

        print("Validation Loss:", self.val_loss[-1])
        print("Validation Accuracy:", self.val_accuracy[-1])
        print("Validation IoU Class:", self.val_iou_class[-1])
        print("Validation IoU Category:", self.val_iou_category[-1])
        # self.train_loss = self.train_loss + [loss.cpu().detach().numpy() for sublist in self.checkpoint0["loss"]["train"] for loss in sublist]
        # self.train_accuracy = self.train_accuracy + [acc.cpu().detach().numpy() for batch_acc in self.checkpoint0["accuracy"]["train"] for acc in batch_acc]
        # self.train_iou_class = self.train_iou_class + [iou.cpu().detach().numpy() for batch_iou in self.checkpoint0["iou_class"]["train"] for iou in batch_iou]
        # self.train_iou_category = self.train_iou_category + [iou.cpu().detach().numpy() for batch_iou in self.checkpoint0["iou_category"]["train"] for iou in batch_iou]

        # self.val_loss = self.val_loss + [loss.cpu().detach().numpy() for sublist in self.checkpoint0["loss"]["val"] for loss in sublist]
        # self.val_accuracy = self.val_accuracy + [acc.cpu().detach().numpy() for batch_acc in self.checkpoint0["accuracy"]["val"] for acc in batch_acc]
        # self.val_iou_class = self.val_iou_class + [iou.cpu().detach().numpy() for batch_iou in self.checkpoint0["iou_class"]["val"] for iou in batch_iou]
        # self.val_iou_category = self.val_iou_category + [iou.cpu().detach().numpy() for batch_iou in self.checkpoint0["iou_category"]["val"] for iou in batch_iou]
    
    def plot(self, task, y, label, name):

        fig, ax = plt.subplots()
        
        if task=="train":
            custom_x_ticks = [self.train_data/self.batch_size*i for i in range(1, self.epochs+1)]
            custom_x_labels = [i for i in range(1, self.epochs+1)]
            x = [i for i in range(len(y))]
            ax.set_xticks(custom_x_ticks)
            ax.set_xticklabels(custom_x_labels)
        if task=="val":
            # custom_x_ticks = [self.val_data/self.batch_size*i for i in range(1, self.epochs+1)]
            # custom_x_labels = [i for i in range(1, self.epochs+1)]

            x = [i for i in range(1, self.epochs+1)]

        ax.plot(x, y, label=label)

        ax.set_xlabel("Epochs")
        ax.set_ylabel(label)

        if not os.path.exists(f"./outputs/results/{self.model_name}"):
            os.makedirs(f"./outputs/results/{self.model_name}")

        fig.savefig(f"./outputs/results/{self.model_name}/{name}")
    
    def save_all(self):
        self.plot("train", self.train_loss, "Train - Cross Entropy Loss", "CrossEntropyLoss_Train.png")
        self.plot("train", self.train_accuracy, "Train - Accuracy", "Accuracy_Train.png")
        self.plot("train", self.train_iou_class, "Train - IoU Class", "IoU_Class_Train.png")
        self.plot("train", self.train_iou_category, "Train - IoU Category", "IoU_Category_Train.png")

        self.plot("val", self.val_loss, "Validation - Cross Entropy Loss", "CrossEntropyLoss_Validation.png")
        self.plot("val", self.val_accuracy, "Validation - Accuracy", "Accuracy_Validation.png")
        self.plot("val", self.val_iou_class, "Validation - IoU Class", "IoU_Class_Validation.png")
        self.plot("val", self.val_iou_category, "Validation - IoU Category", "IoU_Category_Validation.png")

if __name__ == "__main__":


    # checkpoint = "ResNet101_bs16_512x1024"
    # batch_size = 16
    # epoch = 10
    try:
        checkpoint = sys.argv[1]
        batch_size = int(sys.argv[2])
        epoch = int(sys.argv[3])
        print(type(checkpoint), checkpoint)
        print(type(batch_size), batch_size)
        print(type(epoch), epoch)
        
        ssb = SemSegBoard(checkpoint, batch_size, epoch)
        ssb.save_all()
    except Exception as e:
        print(e)
