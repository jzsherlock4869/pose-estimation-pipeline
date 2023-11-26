import albumentations as A
from albumentations.pytorch import ToTensorV2

def train_augment(size):

    aug = A.Compose([
            A.Resize(size, size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
                ),
            ToTensorV2(p=1.0),
        ], p=1.)

    return aug


def val_augment(size):

    aug = A.Compose([
            A.Resize(size, size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
                ),
            ToTensorV2(p=1.0),
        ], p=1.)

    return aug

