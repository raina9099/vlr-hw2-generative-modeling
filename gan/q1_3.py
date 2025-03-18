import argparse
import os
from utils import get_args

import torch

from networks import Discriminator, Generator
import torch.nn.functional as F
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    ##################################################################
    # TODO 1.3: Implement GAN loss for discriminator.
    # Do not use discrim_interp, interp, lamb. They are placeholders
    # for Q1.5.
    ##################################################################
    # Discriminator should maximize log(D(x)) + log(1-D(G(z)))
    # Which is equivalent to minimizing the negative of that expression
    # loss = -(torch.mean(torch.log(discrim_real)) + torch.mean(torch.log(1 - discrim_fake)))
    # TODO: Refactor
    B = discrim_real.size(dim = 0)
    criterion = torch.nn.BCEWithLogitsLoss()
    real_label = torch.ones(B,1).cuda()
    fake_label = torch.zeros(B,1).cuda()
    loss1 = criterion(discrim_real,real_label)
    loss2 = criterion(discrim_fake,fake_label)
    loss = loss1 + loss2
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


def compute_generator_loss(discrim_fake):
    ##################################################################
    # TODO 1.3: Implement GAN loss for the generator.
    ##################################################################
    # Generator should maximize log(D(G(z))) 
    # Which is equivalent to minimizing -log(D(G(z)))
    # loss = -torch.mean(torch.log(discrim_fake))
    # TODO: Refactor
    criterion = torch.nn.BCEWithLogitsLoss()
    B = discrim_fake.size(dim = 0)
    real_label = torch.ones(B,1).cuda()
    loss = criterion(discrim_fake,real_label)
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
