from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import torch
import transformers
import typer
from accelerate import Accelerator
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

import utils
from data import dataloader
from models import segformer
from models.segformer import get_feature_extractor
from schemas import Config


def get_clahe(config: Config):
    clip_limit = config.dataset.augmentation.clahe.clip_limit
    height = config.dataset.augmentation.clahe.grid_height
    width = config.dataset.augmentation.clahe.grid_width
    return A.CLAHE(
        always_apply=False,
        p=1.0,
        clip_limit=(clip_limit, clip_limit),
        tile_grid_size=(height, width),
    )


def is_dark(image: np.ndarray, threshold=127):
    return True if np.mean(image) < threshold else False


root_dir = Path("/home/artem/datasets/rzd_public/")
id2label = {0: "background", 1: "secondary_rails",
            2: "main_rails", 3: "train_car"}
label2id = {"background": 0, "secondary_rails": 1,
            "main_rails": 2, "train_car": 3}
id2color = {0: [0, 0, 0], 1: [6, 6, 6], 2: [7, 7, 7], 3: [10, 10, 10]}
id2color_rgb = {0: [255, 255, 255], 1: [
    0, 0, 255], 2: [255, 0, 0], 3: [0, 255, 0]}
device = "cuda"


def save_logits(config: Config, logits, filename: str, prefix: float):
    logits = logits.numpy()
    exp_name = config.exp_name
    save_dir = (
        Path()
        .absolute()
        .joinpath("predictions")
        .joinpath(exp_name)
        .joinpath(f"predictions_{prefix}")
    )
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir.joinpath(f"{filename}.npy")
    np.save(str(save_path), logits)




def tta_models(pixel_values, model):

    pixel_values_flipped = torch.flip(pixel_values, (3,))

    pixel_values_rot = torch.rot90(pixel_values, 2, [2, 3])

    outputs = model(pixel_values=pixel_values)

    outputs_flipped = model(pixel_values=pixel_values_flipped)

    outputs_rot = model(pixel_values=pixel_values_rot)

    outputs_flipped.logits = torch.flip(
        outputs_flipped.logits, (3,))

    outputs_rot.logits = torch.rot90(outputs_rot.logits, 2, [2,3])

    logits_rot = outputs_rot.logits.cpu()

    logits_flipped = outputs_flipped.logits.cpu()

    logits = outputs.logits.cpu()

    return (logits_flipped+logits+logits_rot)/3


def simple_inference(config: Config, model, scale):
    test_dir = Path(config.dataset.test_dir)
    # test_dir=root_dir
    for image_path in tqdm(list(test_dir.glob("*"))):

        with torch.no_grad():
            image = Image.open(str(image_path)).convert("RGB")

            if scale is not None:
                # print(min(int(image.size[0]*scale), int(image.size[1]*scale)))
                config.dataset.input_size = min(
                    int(image.size[0] * scale), int(image.size[1] * scale)
                )

            np_image = np.array(image)
            feature_extractor = get_feature_extractor(config=config)
            # prepare the image for the model
            encoding = feature_extractor(image, return_tensors="pt")
            pixel_values = encoding.pixel_values.to(device)
            # forward pass

            mean_logits = tta_models(
                model=model, pixel_values=pixel_values)

            prefix = scale if scale is not None else config.dataset.input_size
            save_logits(
                config=config,
                logits=mean_logits,
                filename=image_path.name,
                prefix=prefix,
            )


def run_inference(config: Config, checkpoint_path: str, scale: float):
    print("Load Feature Extractor...")
    # config.dataset.input_size = 1340
    # feature_extractor = segformer.get_feature_extractor(config)
    print("Done.")

    # print("Preparing train and val dataloaders...")
    # test_dataloader = dataloader.get_test_dataloader(config, feature_extractor)
    # print("Done.")
    accelerator = Accelerator(fp16=True)
    device = accelerator.device
    print("Load model...")
    model = segformer.get_model(config)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cuda")["state_dict"]
    model.load_state_dict(checkpoint)
    print("Done.")

    # model, test_dataloader = accelerator.prepare(model, test_dataloader)
    print("Done.")
    simple_inference(config=config, model=model, scale=scale)
    # inference(model=model, dataloader=test_dataloader, config=config)


def main(
    cfg: Path = typer.Option("configs/baseline.yml", help="Путь до конфига"),
    checkpoint_path: Path = typer.Option("", help="Путь до чекпоинта модели"),
    scale: Optional[float] = typer.Option(None, help="Scale image"),
):
    config: Config = OmegaConf.load(cfg)
    utils.set_global_seed(config)
    utils.prepare_predictions_dir(config)
    run_inference(config, checkpoint_path, scale)


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        typer.secho("Программа завершена", fg=typer.colors.BRIGHT_GREEN)
