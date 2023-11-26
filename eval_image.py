from dataset import CityscapesLoader
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from PIL import Image
from models.unet import UNet
import numpy as np

def evaluate_model(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

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

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i == 2:
                break

            batch_images, batch_labels = batch[0], batch[1]

            eval_images = batch_images.to(device)
            eval_labels = batch_labels.to(device)

            eval_pred = model(eval_images)

            prediction = F.interpolate(eval_pred, size=(1024, 2048), mode="bilinear", align_corners=True)
            prediction = prediction.data.max(1)[1].cpu().numpy()
            decoded_pred = eval_data.decode_segmap(prediction[0])
            img1 = Image.fromarray(decoded_pred.astype(np.uint8))
            img1.save(f"prediction_5_6_{i}.png")

            ground_truth = eval_labels.data.cpu().numpy()
            decode_gt = eval_data.decode_segmap(ground_truth[0])

            img2 = Image.fromarray(decode_gt.astype(np.uint8))
            img2.save(f"target_{i}.png")

model = UNet(19)
model_name = "UNet001_bs5_512x1024"
epoch = 6
checkpoint = torch.load(f"./outputs/checkpoints/{model_name}/{model_name}__e{epoch}")
print("Time:", checkpoint["total_time"]/60, "min")
model.load_state_dict(checkpoint['model_state_dict'])

evaluate_model(model)