import albumentations as A

from schemas import Config

train_transform = A.Compose(
    [
                A.Rotate(limit=90, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # A.CLAHE(always_apply=False, p=1.0, clip_limit=(16, 19), tile_grid_size=(28, 30)),
                # A.ElasticTransform(alpha=120, sigma=120*0.05, alpha_affine=120),
                # A.Transpose(always_apply=False, p=0.5),
                A.RandomCropFromBorders(always_apply=False, p=0.5, crop_left=0.1, crop_right=0.1, crop_top=0.1, crop_bottom=0.1),
                A.RandomContrast(always_apply=False, p=0.5, limit=(-0.07, 0.13))
            ]
            )  


val_transform = A.Compose([])


def get_train_transform(config: Config):
    if config.dataset.with_augs:
        return train_transform
    return val_transform
