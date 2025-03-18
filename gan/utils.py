import argparse
import torch
from cleanfid import fid
from matplotlib import pyplot as plt
import torchvision.utils as vutils


def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    ##################################################################
    # TODO: 1.2: Generate and save out latent space interpolations.
    # 1. Generate 100 samples of 128-dim vectors. Do so by linearly
    # interpolating for 10 steps across each of the first two
    # dimensions between -1 and 1. Keep the rest of the z vector for
    # the samples to be some fixed value (e.g. 0).
    # 2. Forward the samples through the generator.
    # 3. Save out an image holding all 100 samples.
    # Use torchvision.utils.save_image to save out the visualization.
    ##################################################################
    # Create a meshgrid for the first two dimensions
    x = torch.linspace(-1, 1, 10)
    y = torch.linspace(-1, 1, 10)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    
    # Create the latent vectors (100 samples of 128-dim vectors)
    z = torch.zeros(100, 128).cuda()  
    z[:, 0] = x_grid.flatten()  
    z[:, 1] = y_grid.flatten()  
    
    # Generate images
    images = gen.forward_given_samples(z)
    
    # Save the grid of images (normalize to [0,1] range for saving)
    vutils.save_image(images, path, normalize=True, nrow=10)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()
    return args
