from dataset import CityscapesLoader

if __name__ == "__main__":

    cityscapes_root = "../../../../projects/vc/data/ad/open/Cityscapes/"

    print("Loading dataset...")

    train_data = CityscapesLoader(
        root = cityscapes_root, 
        split = 'train',
        # shape=train_resize
    )

    val_data = CityscapesLoader(
        root = cityscapes_root, 
        split = 'val',
        # shape=val_resize
    )

    test_data = CityscapesLoader(
        root = cityscapes_root, 
        split = 'test',
        # shape=val_resize
    )
