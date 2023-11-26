import os
import torchvision
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import sklearn.metrics as skm
import numpy as np
import time
from tqdm import tqdm
from matplotlib import pyplot as plt 
from PIL import Image
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

from dataset import CityscapesLoader
from models.resnet import ResNet
from models.unet import UNet

def cross_entropy2d(input, target, gamma=False, weight=None, reduction="mean"):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    ce_loss = F.cross_entropy(
        input, target, weight=weight, reduction=reduction, ignore_index=250
    )
    if gamma:
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss
        loss = focal_loss
    else:
        loss = ce_loss

    return loss

class SemanticSegmentation():

    def __init__(self, model, optimizer, train_data, val_data, epochs, batch_size, num_workers, learning_rate, model_path=None, gamma=False):
        print("----------------------------------------")
        print("Initializing Model")
        print("Batch Size:", batch_size, "\nEpochs:", epochs, "\nLearning Rate:", learning_rate, "\nOptimizer:", optimizer)
        print("Cross Entropy Loss", f"w/Focal Loss, gamma: {gamma}" if gamma else "")
        print("----------------------------------------")
        self.model = model
        # self.optimizer = optimizer
        self.epochs = epochs
        self.start_epoch = 0
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.train_loader = DataLoader(
            train_data,
            batch_size = batch_size,
            shuffle = True,
            num_workers = num_workers,
        )
        self.val_loader = DataLoader(
            val_data,
            batch_size = batch_size,
            num_workers = num_workers,
        )
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device:", self.device, "//", torch.cuda.get_device_name(0), "\n----------------------------------------")
        
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.loss = {"train": [], "val": [], "test": []}
        self.accuracy = {"train": [], "val": [], "test": []}
        self.iou_class = {"train": [], "val": [], "test": []}
        self.iou_category = {"train": [], "val": [], "test": []}

        self.total_time = 0

        if model_path:
            checkpoint = torch.load(model_path)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint["epoch"]

            self.loss = checkpoint["loss"]
            self.accuracy = checkpoint["accuracy"]
            self.iou_class = checkpoint["iou_class"]
            self.iou_category = checkpoint["iou_category"]

            self.total_time = checkpoint["total_time"]

            print("Continuing training on epoch", self.start_epoch+1)
    
    def calculate_metrics(self, pred, labels):

        n, c, h, w = pred.size()
        nt, ht, wt = labels.size()

        if h != ht and w != wt: 
            pred = F.interpolate(pred, size=(ht, wt), mode="bilinear", align_corners=True)

        # Considering predictions with highest scores for each pixel among 19 classes
        pred_labels = pred.data.max(1)[1]                                     
        gt_labels = labels

        pred_labels = pred_labels.flatten()
        gt_labels = gt_labels.flatten()

        category_ids = [0, 0, 3, 3, 3, 4, 4, 4, 5, 5, 6, 1, 1, 2, 2, 2, 2, 2, 2]
        category_map = dict(zip(range(19), category_ids))
        pred_catgories = torch.tensor(np.array([category_map[label] for label in pred_labels.data.cpu().numpy()])).to(self.device)
        gt_categories = torch.tensor(np.array([category_map[label] if not label==250 else 250 for label in gt_labels.data.cpu().numpy()])).to(self.device)

        amc = MulticlassAccuracy(num_classes=19+1, average="micro", ignore_index=250).to(self.device)
        jimc_class = MulticlassJaccardIndex(num_classes=19+1, average="micro", ignore_index=250).to(self.device)
        jimc_category = MulticlassJaccardIndex(num_classes=7+1, average="micro", ignore_index=250).to(self.device)

        accuracy = amc(pred_labels, gt_labels)
        IoU_class = jimc_class(pred_labels, gt_labels)
        IoU_category = jimc_category(pred_catgories, gt_categories)

        # print(IoU_class, IoU_category)

        return accuracy, IoU_class, IoU_category

    def train(self, epoch):
        self.model.train()

        batch_loss = []
        batch_accuracy = []
        batch_iou_class = []
        batch_iou_category = []

        tqdm_loader = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.epochs}]", unit="iter")
        for i, batch in enumerate(tqdm_loader):

            batch_images, batch_labels = batch[0], batch[1]

            images = batch_images.to(self.device)
            labels = batch_labels.to(self.device)

            pred = self.model(images)

            loss = cross_entropy2d(pred, labels, self.gamma)

            accuracy, IoU_class, IoU_category = self.calculate_metrics(pred, labels)

            batch_loss.append(loss)
            batch_accuracy.append(accuracy)
            batch_iou_class.append(IoU_class)
            batch_iou_category.append(IoU_category)


            self.optimizer.zero_grad()      # Sets the gradient to zero for all the parameters before performing the backprop
            loss.backward()                 # Calculates the new gradients of the loss
            self.optimizer.step()           # Updates the model's parameters with the given optimizer

            tqdm_loader.set_postfix(ordered_dict={
                "Loss": loss.item(), 
                "Accuracy": accuracy.item(), 
                "IoU_Class": IoU_class.item(), 
                "IoU_Category": IoU_category.item()
            }, refresh=True)

        self.loss["train"].append(batch_loss)
        self.accuracy["train"].append(batch_accuracy)
        self.iou_class["train"].append(batch_iou_class)
        self.iou_category["train"].append(batch_iou_category)
    
    def validate(self, epoch):
        self.model.eval()

        batch_loss = []
        batch_accuracy = []
        batch_iou_class = []
        batch_iou_category = []

        with torch.no_grad():
            tqdm_loader = tqdm(self.val_loader, desc=f"Epoch [{epoch+1}/{self.epochs}]", unit="iter")
            for batch in tqdm_loader:

                batch_images, batch_labels = batch[0], batch[1]

                val_images = batch_images.to(self.device)
                val_labels = batch_labels.to(self.device)

                val_pred = self.model(val_images)

                loss = cross_entropy2d(val_pred, val_labels, self.gamma)
                
                accuracy, IoU_class, IoU_category = self.calculate_metrics(val_pred, val_labels)

                batch_loss.append(loss)
                batch_accuracy.append(accuracy)
                batch_iou_class.append(IoU_class)
                batch_iou_category.append(IoU_category)

                tqdm_loader.set_postfix(ordered_dict={
                    "Loss": loss.item(), 
                    "Accuracy": accuracy.item(), 
                    "IoU_Class": IoU_class.item(), 
                    "IoU_Category": IoU_category.item()
                }, refresh=True)
                
        self.loss["val"].append(batch_loss)
        self.accuracy["val"].append(batch_accuracy)
        self.iou_class["val"].append(batch_iou_class)
        self.iou_category["val"].append(batch_iou_category)
    
    def evaluate_model(self, model, model_name):
        model.to(self.device)
        cityscapes_root = "../../../../projects/vc/data/ad/open/Cityscapes/"

        eval_data = CityscapesLoader(
            root=cityscapes_root, 
            split="val",
        )

        eval_loader = DataLoader(
            eval_data,
            batch_size=1,
            num_workers=1,
        )
        
        model.eval()

        batch_loss = []
        batch_accuracy = []
        batch_iou_class = []
        batch_iou_category = []

        with torch.no_grad():
            tqdm_loader = tqdm(eval_loader, desc=f"Test", unit="iter")
            for i, batch in enumerate(tqdm_loader):

                batch_images, batch_labels = batch[0], batch[1]

                eval_images = batch_images.to(self.device)
                eval_labels = batch_labels.to(self.device)

                eval_pred = model(eval_images)


                if i % 100 == 0:

                    prediction = F.interpolate(eval_pred, size=(1024, 2048), mode="bilinear", align_corners=True)
                    prediction = prediction.data.max(1)[1].cpu().numpy()
                    decoded_pred = eval_data.decode_segmap(prediction[0])
                    img1 = Image.fromarray(decoded_pred.astype(np.uint8))
                    img1.save(f"./outputs/results/{model_name[:-1]}/prediction_{i}.png")

                    # print("PRED SHAPE", eval_labels.shape)
                    ground_truth = eval_labels.data.cpu().numpy()
                    decode_gt = eval_data.decode_segmap(ground_truth[0])
                    
                    # print("TARGET SHAPE", decode_gt.shape)
                    img2 = Image.fromarray(decode_gt.astype(np.uint8))
                    img2.save(f"./outputs/results/{model_name[:-1]}/target_{i}.png")
                

                loss = cross_entropy2d(eval_pred, eval_labels, self.gamma)
                
                accuracy, IoU_class, IoU_category = self.calculate_metrics(eval_pred, eval_labels)

                batch_loss.append(loss.item())
                batch_accuracy.append(accuracy.item())
                batch_iou_class.append(IoU_class.item())
                batch_iou_category.append(IoU_category.item())

                tqdm_loader.set_postfix(ordered_dict={
                    "Loss": loss.item(), 
                    "Accuracy": accuracy.item(), 
                    "IoU_Class": IoU_class.item(), 
                    "IoU_Category": IoU_category.item()
                }, refresh=True)
        
        loss, accuracy, iou_class, iou_category = sum(batch_loss)/len(batch_loss), sum(batch_accuracy)/len(batch_accuracy), sum(batch_iou_class)/len(batch_iou_class), sum(batch_iou_category)/len(batch_iou_category)
        print("Loss:", loss, "\nAccuracy:", accuracy, "\nIoU_class:", iou_class, "\nIoU_category:", iou_category)

        return batch_loss, batch_accuracy, batch_iou_class, batch_iou_class

    def run(self, check_name="Baseline"):

        if not os.path.exists(f"./outputs/checkpoints/{check_name[:-1]}"):
            os.makedirs(f"./outputs/checkpoints/{check_name[:-1]}")

        for epoch in range(self.start_epoch, self.epochs):

            t1 = time.time()
            self.train(epoch)
            self.validate(epoch)   
            t2 = time.time()

            # print(f"Epoch {epoch} took: ", t2-t1, " unit time")
            self.total_time += t2-t1
        
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss,
                "accuracy": self.accuracy,
                "iou_class": self.iou_class,
                "iou_category": self.iou_category,
                "total_time": self.total_time,
            }, f"./outputs/checkpoints/{check_name[:-1]}/{check_name}_e{epoch+1}")

if __name__ == "__main__":

    cityscapes_root = "../../../../projects/vc/data/ad/open/Cityscapes/"
    
    # // Hyperparameters //
    epochs = 10
    batch_size = 5
    num_workers = 1
    learning_rate = 1e-3
    train_resize = (512, 1024)
    val_resize = (512, 1024)
    # train_resize = (1024, 2048)
    # val_resize = (1024, 2048)
    num_classes = 19
    optimizer="Adam"
    gamma = 2 # For focal loss
    model_name = f"ResNet50_bs{batch_size}_{train_resize[0]}x{train_resize[1]}_"
    # model_path=f"./outputs/checkpoints/{model_name[:-1]}/{model_name}_e15"
     
    torch.cuda.empty_cache()

    backbone = torchvision.models.resnet50(weights="DEFAULT")
    # backbone = torchvision.models.resnet101(weights="DEFAULT")
    
    num_features = backbone.fc.in_features
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

    model = ResNet(backbone, num_features, 19) 

    # model = UNet(num_classes)

    print("Loading dataset...")
    train_data = CityscapesLoader(
        root = cityscapes_root, 
        split = 'train',
        shape=train_resize
    )

    val_data = CityscapesLoader(
        root = cityscapes_root, 
        split = 'val',
        shape=val_resize
    )

    SemSeg = SemanticSegmentation(
        model, optimizer, train_data, val_data, epochs, batch_size, num_workers, learning_rate, None, None)
    # SemSeg.run(model_name)

    # epoch=16
    # checkpoint = torch.load(f"./outputs/checkpoints/{model_name[:-1]}/{model_name}_e{epoch}")
    # print("Time:", checkpoint["total_time"]/60, "min")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # eval_metrics = SemSeg.evaluate_model(model, model_name)

    # if True:
    #     # print("Loss:", eval_metrics[0])
    #     # print("Acurracy", eval_metrics[1])
    #     # print("IoU Class", eval_metrics[2])
    #     # print("IoU Category", eval_metrics[3])
    #     # plt.plot(eval_metrics[0], linestyle='-', color='red', label='Cross Entropy Loss')
    #     plt.plot(eval_metrics[1], linestyle='-', color='blue', label='Accuracy')
    #     plt.plot(eval_metrics[2], linestyle='-', color='green', label='IoU Class')
    #     plt.plot(eval_metrics[3], linestyle='-', color='yellow', label='IoU Category')

    #     plt.xlabel('X-axis')
    #     plt.ylabel('Y-axis')
    #     plt.title('Metrics of Model')

    #     plt.legend()
    #     plt.savefig('Evaluation_Metrics.png')