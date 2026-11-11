"""Deterministic epoch permutations and isolated dataloader seeding."""
import random

import numpy as np
import torch
from torch.utils.data import Sampler


def seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class ResumableEpochSampler(Sampler):
    """Accelerate shards a shared permutation; offsets count global microbatches."""
    def __init__(self, dataset, seed, batch_size, num_processes):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = self.epoch_offset = self.resume_batch_offset = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch):
        self.epoch_offset = int(epoch)

    def set_resume_batch_offset(self, batch):
        self.resume_batch_offset = int(batch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch + self.epoch_offset)
        order = torch.randperm(len(self.dataset), generator=generator).tolist()
        if self.num_processes > 1:
            global_batch = self.batch_size * self.num_processes
            padding = (-len(order)) % global_batch
            if padding:
                order += (order * ((padding + len(order) - 1) // len(order)))[:padding]
        if self.epoch == 0:
            order = order[self.resume_batch_offset * self.batch_size * self.num_processes:]
        return iter(order)

    def __len__(self):
        length = len(self.dataset)
        global_batch = self.batch_size * self.num_processes
        return length if self.num_processes == 1 else ((length + global_batch - 1) // global_batch) * global_batch
