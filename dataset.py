import torch
from torch.utils import data
import os
from PIL import Image
import numpy as np

def get_all_image_paths(rootdir):
    image_paths = []
    cities = os.listdir(rootdir)
    for city in cities:
        images = os.listdir(os.path.join(rootdir, city))
        for image in images:
            image_paths.append(os.path.join(rootdir, city, image))
    return image_paths

class CityscapesLoader(data.Dataset):

    def __init__(self, root, split="train", shape=(1024, 2048)):
        self.root = root
        self.split = split
        self.width = shape[1]
        self.height = shape[0]
        # self.is_transform = is_transform
        self.n_classes = 19
        # self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit_trainvaltest", "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine_trainvaltest", "gtFine", self.split)

        self.files[split] = get_all_image_paths(self.images_base)

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_map = dict(zip(self.valid_classes, range(19)))
        
        # print("CLASSMAP", self.class_map)
        # these are 19 + 1; "unlabelled" is extra
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]
        
        # for void_classes; useful for loss function
        self.ignore_index = 250

        colors = [  
            # [  0,   0,   0],
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [0, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ]

        self.label_colors = dict(zip(range(19), colors))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))
        
        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()

        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        img = Image.open(img_path)
        img = img.resize((self.width, self.height), Image.NEAREST)
        img = np.array(img, dtype=np.uint8)
        img = img.astype(np.float64)
        img = np.transpose(img, (2, 0, 1)) # NHWC -> NCHW

        lbl = Image.open(lbl_path)
        lbl = lbl.resize((self.width, self.height), Image.NEAREST)
        lbl = np.array(lbl, dtype=np.uint8)
        lbl = self.encode_segmap(lbl)
        lbl = lbl.astype(int)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl
    
    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colors[l][0]
            g[temp == l] = self.label_colors[l][1]
            b[temp == l] = self.label_colors[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r #/ 255.0
        rgb[:, :, 1] = g #/ 255.0
        rgb[:, :, 2] = b #/ 255.0
        return rgb
    
    def encode_segmap(self, mask):
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask