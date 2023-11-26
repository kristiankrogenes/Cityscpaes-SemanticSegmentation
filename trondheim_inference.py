import os
from PIL import Image
from models.unet import UNet
import numpy as np
import torch 
from dataset import CityscapesLoader

if __name__ == "__main__":

    cityscapes_root = "../../../../projects/vc/data/ad/open/Cityscapes/"

    DataLoader = CityscapesLoader(
        root=cityscapes_root, 
        split="val",
    )

    root = "./trondheim"

    trondheim_images = os.listdir(root)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device, "//", torch.cuda.get_device_name(0), "\n----------------------------------------")

    model = UNet(19)
    model_name = "UNet001_bs5_1024x2048"
    epoch = 20
    checkpoint = torch.load(f"./outputs/checkpoints/{model_name}/{model_name}__e{epoch}")
    print("Time:", checkpoint["total_time"]/60, "min")
    model.load_state_dict(checkpoint['model_state_dict'])

    for i, trondheim_image in enumerate(trondheim_images):
        print("Image 1")

        image = Image.open(f"{root}/{trondheim_image}")
        image = np.array(image, dtype=np.uint8)
        image = image.astype(np.float64)
        image = np.transpose(image, (2, 0, 1)) # NHWC -> NCHW
        image = torch.from_numpy(np.array([image])).float()

        image.to(device)

        pred = model(image)

        prediction = pred.data.max(1)[1].cpu().numpy()
        decoded_pred = DataLoader.decode_segmap(prediction[0])
        prediction_image = Image.fromarray(decoded_pred.astype(np.uint8))
        prediction_image.save(f"./trondheim/trondheim_prediction_{i}.png")


        # new_image = Image.fromarray(image[0].astype(np.uint8))
        # new_image.save(f"./{trondheim_image}")

    # eval_pred = model(eval_images)

    # prediction = F.interpolate(eval_pred, size=(1024, 2048), mode="bilinear", align_corners=True)
    # prediction = prediction.data.max(1)[1].cpu().numpy()
    # decoded_pred = eval_data.decode_segmap(prediction[0])
    # img1 = Image.fromarray(decoded_pred.astype(np.uint8))
    # img1.save(f"prediction_5_6_{i}.png")

