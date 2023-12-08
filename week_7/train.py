import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import setup_logging, get_data, save_images
from ddpm import Diffusion
from modules import UNet_conditional
import logging
from torch.utils.tensorboard import SummaryWriter
import numpy as np

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet_conditional(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(log_dir=os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(
            model, n=images.shape[0], labels=labels, cfg_scale=3
        )
        save_images(
            sampled_images, os.path.join("results", args.run_name, f"{epoch}.png")
        )
        torch.save(
            model.state_dict(),
            os.path.join("models", args.run_name, f"checkpoint_epoch_{epoch}.pt"),
        )


def launch():
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs = 500
    args.batch_size = 8
    args.image_size = 64
    args.num_classes = 3
    args.dataset_path = "PokemonData"
    args.device = "cuda"
    args.lr = 1e-4
    train(args)


if __name__ == "__main__":
    launch()
