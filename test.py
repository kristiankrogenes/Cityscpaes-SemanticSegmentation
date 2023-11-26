import torch

from dataset import CityscapesLoader
from torch.utils.data import DataLoader
from tqdm import tqdm
import sklearn.metrics as skm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device, "//", torch.cuda.get_device_name(0), "\n---------------------")
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)

from torchmetrics.classification import MulticlassJaccardIndex

arr = np.array([[
    [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], 
    [[10, 11, 12], [10, 11, 12], [10, 11, 12], [10, 11, 12], [10, 11, 12]], 
    [[20, 21, 22], [20, 21, 22], [20, 21, 22], [20, 21, 22], [20, 21, 22]]
]])

arr_transposed = np.transpose(arr, (0, 3, 1, 2))
print(arr_transposed)

# pred = torch.tensor([[[1, 0, 3, 2, 0, 1, 2, 1, 0, 2], [1, 0, 3, 2, 0, 1, 2, 1, 0, 2], [1, 0, 3, 2, 0, 1, 2, 1, 0, 2], [1, 0, 3, 2, 0, 1, 2, 1, 0, 2]], [[1, 0, 3, 2, 0, 1, 2, 1, 0, 2], [1, 0, 3, 2, 0, 1, 2, 1, 0, 2], [1, 0, 3, 2, 0, 1, 2, 1, 0, 2], [1, 0, 3, 2, 0, 1, 2, 1, 0, 2]]])
# gtla = torch.tensor([[[1, 3, 3, 1, 5, 5, 4, 4, 2, 2], [1, 3, 3, 1, 5, 5, 4, 4, 1, 2], [1, 3, 3, 1, 5, 5, 4, 4, 3, 2], [1, 3, 3, 1, 5, 5, 4, 4, 2, 2]], [[1, 3, 3, 1, 5, 5, 4, 4, 0, 2], [1, 3, 3, 1, 5, 5, 4, 4, 0, 2], [1, 3, 3, 1, 5, 5, 4, 4, 0, 2], [1, 3, 3, 1, 5, 5, 4, 4, 0, 2]]])
# pred = pred.to(device)
# gtla = gtla.to(device)


# jaccard = MulticlassJaccardIndex(num_classes=10, average="micro").to(device)
# score = jaccard(pred.flatten(), gtla.flatten())

# p = pred.data.cpu().numpy()                                            
# g = gtla.data.cpu().numpy()
# i = skm.jaccard_score(g.flatten(), p.flatten(), average='micro')
# print("SCORE", score, i)

# a = [1, 2, 3, 4]
# b= [5, 6]
# print(a+b)

# cityscapes_root = "../../../../projects/vc/data/ad/open/Cityscapes/"
# batch_size = 1
# num_workers = 1

# train_data = CityscapesLoader(
#     root = cityscapes_root, 
#     split = 'train'
# )

# train_loader = DataLoader(
#     train_data,
#     batch_size = batch_size,
#     shuffle = True,
#     num_workers = num_workers,
# )