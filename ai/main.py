import torch
from torchvision.utils import save_image
from torch.optim import Adam
from pathlib import Path

from ai.data import fashion
from ai.data import main as D
from ai.model.layers import Unet
from ai.model import diffusion


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def get_noisy_image(x_start, t):
    # add noise
    x_noisy = diffusion.q_sample(x_start, t=t)

    # turn back into PIL image
    noisy_image = D.reverse_transform(x_noisy.squeeze())

    return noisy_image


def main():
    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)
    image_size = 28
    channels = 1
    # batch_size = 128
    save_and_sample_every = 200

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,)
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)
    dataloader = fashion.get_fashion_dataset()

    epochs = 5

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()

            loss = diffusion.p_losses(model, batch, t, loss_type="huber")

            if step % 100 == 0:
                print("Loss:", loss.item(), "step:", step)

            loss.backward()
            optimizer.step()

            # save generated images
            if step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                batches = num_to_groups(4, batch_size)
                all_images_list = list(map(lambda n: diffusion.sample(model, image_size, batch_size=n, channels=channels), batches))
                all_images = torch.cat(all_images_list[0], dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow=6)

    return model


if __name__ == '__main__':
    main()
