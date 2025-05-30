import glob
import os
import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import numpy as np
from tqdm import tqdm, trange
import math
import random

import argparse

parser = argparse.ArgumentParser()

# -i 0 --createnew
parser.add_argument("-i", "--iteration", help="Database name", type=int, default=0)
parser.add_argument("--createnew", help="Create new network", action="store_true")
args = parser.parse_args()

GAME_NAME = "connect4"
CHECKPOINT_PATH = os.path.join("data", "checkpoint")
DATASET_PATH = os.path.join("data", "dataset")

LOSS_C_V = 1.7   # coefficient of value loss

TRAIN_BATCH_SIZE = 1024  # This generally should be as big as can fit on your gpu.
TRAIN_SAMPLE_RATE = 1  # Note: If the game has a high number of symetries generated, this number should likely get lowered.

# To decide on the following numbers, I would advise graphing the equation: scalar*(1+beta*(((iter+1)/scalar)**alpha-1)/alpha)
WINDOW_SIZE_ALPHA = 0.5  # This decides how fast the curve flattens to a max
WINDOW_SIZE_BETA = 0.5   # This decides the rough overall slope.
WINDOW_SIZE_SCALAR = 5   # This ends up being approximately first time history doesn't grow

iteration = args.iteration
create_new = args.createnew

def checkpoint_filepath():
    return f"{CHECKPOINT_PATH}/{iteration:04d}-{GAME_NAME}.pt"

from nnarch_connect4 import *
class NNWrapper:
    def __init__(self, args):
        self.args = args
        self.nnet = NNArch(args)
        self.optimizer = optim.SGD(
            self.nnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3
        )

        def lr_lambda(epoch):
            if epoch < 10:  # warm-up
                return 1/3
            else:
                return 1

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )
        self.cv = args.cv
        self.nnet.cuda()

    def train(self, batches, steps_to_train):
        self.nnet.train()

        v_loss = 0
        pi_loss = 0
        current_step = 0
        pbar = tqdm(
            total=steps_to_train, unit="batches", desc="Training NN", leave=False
        )
        past_states = []
        while current_step < steps_to_train:
            for batch in batches:
                if (
                    steps_to_train // 4 > 0
                    and current_step % (steps_to_train // 4) == 0
                    and current_step != 0
                ):
                    # Snapshot model weights
                    past_states.append(dict(self.nnet.named_parameters()))
                if current_step == steps_to_train:
                    break
                canonical, target_vs, target_pis = batch
                canonical = canonical.contiguous().cuda()
                target_vs = target_vs.contiguous().cuda()
                target_pis = target_pis.contiguous().cuda()

                # reset grad
                self.optimizer.zero_grad()

                # forward + backward + optimize
                out_v, out_pi = self.nnet(canonical)
                l_v = self.loss_v(target_vs, out_v)
                l_pi = self.loss_pi(target_pis, out_pi)

                # TODO: maybe add L2-penalty on model parameters?
                total_loss = l_pi + self.cv * l_v
                total_loss.backward()
                self.optimizer.step()

                # record loss and update progress bar.
                pi_loss += l_pi.item()
                v_loss += l_v.item()
                current_step += 1
                pbar.set_postfix(
                    {
                        "v loss": v_loss / current_step,
                        "pi loss": pi_loss / current_step,
                        "total": (self.cv * v_loss + pi_loss) / current_step,
                    }
                )
                pbar.update()

        # Perform expontential averaging of network weights.
        past_states.append(dict(self.nnet.named_parameters()))
        merged_states = past_states[0]
        for state in past_states[1:]:
            for k in merged_states.keys():
                merged_states[k].data = (
                    merged_states[k].data * 0.75 + state[k].data * 0.25
                )
        nnet_dict = self.nnet.state_dict()
        nnet_dict.update(merged_states)
        self.nnet.load_state_dict(nnet_dict)

        self.scheduler.step()
        print(f"Current learn rate={self.scheduler.get_last_lr()}")
        pbar.close()
        return

    def predict(self, canonical, debug=False):
        v, pi = self.process(canonical.unsqueeze(0), debug)
        return v[0], pi[0]

    def process(self, batch, debug=False):
        batch = batch.contiguous().cuda()
        self.nnet.eval()
        with torch.no_grad():
            v, pi = self.nnet(batch, debug=debug)
            res = (torch.exp(v), torch.exp(pi))
        return res

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def sample_loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs, axis=1)

    def sample_loss_v(self, targets, outputs):
        return -torch.sum(targets * outputs, axis=1)

    def sample_loss(self, dataset, size):
        loss = np.zeros(size)
        self.nnet.eval()
        i = 0
        for batch in tqdm(dataset, desc="Calculating Sample Loss", leave=False):
            canonical, target_vs, target_pis = batch
            canonical = canonical.contiguous().cuda()
            target_vs = target_vs.contiguous().cuda()
            target_pis = target_pis.contiguous().cuda()

            out_v, out_pi = self.nnet(canonical)
            l_v = self.sample_loss_v(target_vs, out_v).detach().cpu()
            l_pi = self.sample_loss_pi(target_pis, out_pi).detach().cpu()
            total_loss = self.cv * l_v + l_pi
            for sample_loss in total_loss:
                loss[i] = sample_loss
                i += 1
        return loss

    def save_checkpoint(self, filepath):
        torch.save(
            {
                "state_dict": self.nnet.state_dict(),
                "opt_state": self.optimizer.state_dict(),
                "sch_state": self.scheduler.state_dict(),
                "args": self.args,
            },
            filepath,
        )

        traced_script_module = torch.jit.trace(
            self.nnet.eval(), torch.ones((1,) + INPUT_SIZE).cuda()
        )
        traced_script_module.save(filepath[:-3] + "_traced.pt")

    @staticmethod
    def load_checkpoint(filepath):
        checkpoint = torch.load(filepath)
        net = NNWrapper(checkpoint["args"])
        net.nnet.load_state_dict(checkpoint["state_dict"])
        net.optimizer.load_state_dict(checkpoint["opt_state"])
        net.scheduler.load_state_dict(checkpoint["sch_state"])
        return net


def create_init_net(nnargs):
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    nn = NNWrapper(nnargs)
    nn.save_checkpoint(checkpoint_filepath())
    return nn


def calc_hist_size(i):
    return int(
        WINDOW_SIZE_SCALAR
        * (
            1
            + WINDOW_SIZE_BETA
            * (((i + 1) / WINDOW_SIZE_SCALAR) ** WINDOW_SIZE_ALPHA - 1)
            / WINDOW_SIZE_ALPHA
        ) - 1                            # minus one here because client-server lag
    )


def maybe_save(folder, c, v, p, size, batch, force=False):
    if size >= TRAIN_BATCH_SIZE or (force and size > 0):
        with torch.no_grad():
            c_tensor = c.clone()
            c_tensor.resize_(size, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2])
            v_tensor = v.clone()
            v_tensor.resize_(size, V_SIZE)
            p_tensor = p.clone()
            p_tensor.resize_(size, PI_SIZE)
            torch.save(c_tensor, os.path.join(folder, f"c_{batch:04d}_{size}.pt"))
            torch.save(v_tensor, os.path.join(folder, f"v_{batch:04d}_{size}.pt"))
            torch.save(p_tensor, os.path.join(folder, f"p_{batch:04d}_{size}.pt"))
        return True
    return False


def resample_by_surprise(nn, folder, destination):
    with open(os.path.join(destination, "resample.txt"), "w") as f:
        c_names = sorted(glob.glob(os.path.join(folder, "c_*_*.pt")))
        v_names = sorted(glob.glob(os.path.join(folder, "v_*_*.pt")))
        p_names = sorted(glob.glob(os.path.join(folder, "p_*_*.pt")))
        if len(c_names) == 0 or len(v_names) == 0 or len(p_names) == 0:
            print("Error: No training data found when resampling", folder)
            exit(1)
        if not len(c_names) == len(v_names) == len(p_names):
            print("Error: Dataset size mismatch when resampling", folder)
            exit(1)

        datasets = []
        for j in range(len(c_names)):
            c = torch.load(c_names[j])
            v = torch.load(v_names[j])
            p = torch.load(p_names[j])
            datasets.append(TensorDataset(c, v, p))

        dataset = ConcatDataset(datasets)
        sample_count = len(dataset)
        dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

        loss = nn.sample_loss(dataloader, sample_count)
        total_loss = np.sum(loss)

        i_out = 0
        batch_out = 0
        total_out = 0
        c_out = torch.zeros(
            TRAIN_BATCH_SIZE, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2]
        )
        v_out = torch.zeros(TRAIN_BATCH_SIZE, V_SIZE)
        p_out = torch.zeros(TRAIN_BATCH_SIZE, PI_SIZE)

        for i in trange(sample_count, desc="Resampling Data", leave=False):
            sample_weight = 0.5 + (loss[i] / total_loss) * 0.5 * sample_count
            if math.isnan(sample_weight):
                print("Error: NaN sample weight.")
                exit(1)
            for _ in range(math.floor(sample_weight)):
                c, v, pi = dataset[i]
                c_out[i_out] = c
                v_out[i_out] = v
                p_out[i_out] = pi
                i_out += 1
                total_out += 1
                if maybe_save(destination, c_out, v_out, p_out, i_out, batch_out):
                    i_out = 0
                    batch_out += 1
            if random.random() < sample_weight - math.floor(sample_weight):
                c, v, pi = dataset[i]
                c_out[i_out] = c
                v_out[i_out] = v
                p_out[i_out] = pi
                i_out += 1
                total_out += 1
                if maybe_save(destination, c_out, v_out, p_out, i_out, batch_out):
                    i_out = 0
                    batch_out += 1

        maybe_save(destination, c_out, v_out, p_out, i_out, batch_out, force=True)

        f.write(f"{total_out=}\n")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        exit(1)
    torch.backends.cudnn.benchmark = True

    if create_new:
        print("Creating new network...")
        nn = create_init_net(
            NNArgs(
                v_size=V_SIZE,
                pi_size=PI_SIZE,
                num_channels=NN_CHANNELS,
                depth=NN_DEPTH,
                lr_milestone=NN_LR_MILESTONE,
                lr=NN_LR,
                cv=LOSS_C_V,
            ),
        )
        print("New network created.")
        exit(0)
    else:
        print("Loading checkpoint...")
        nn = NNWrapper.load_checkpoint(checkpoint_filepath())
        print("Checkpoint loaded.")

    current_dataset = os.path.join(DATASET_PATH, f"{iteration:04d}")
    if not os.path.exists(current_dataset):
        print("Error: dataset folder not exist, iteration = ", iteration)
        exit(1)

    if not os.path.exists(os.path.join(current_dataset, "resample.txt")):
        os.rename(current_dataset, current_dataset + "_raw")
        os.makedirs(current_dataset, exist_ok=True)
        resample_by_surprise(nn, current_dataset + "_raw", current_dataset)

    print("Loading training data...")
    hist_size = calc_hist_size(iteration)
    print(f"Current history size: {hist_size}")
    total_size = 0
    datasets = []
    for i in range(max(0, iteration - hist_size), iteration + 1):
        c_names = sorted(glob.glob(os.path.join(DATASET_PATH, f"{i:04d}/c_*_*.pt")))
        v_names = sorted(glob.glob(os.path.join(DATASET_PATH, f"{i:04d}/v_*_*.pt")))
        p_names = sorted(glob.glob(os.path.join(DATASET_PATH, f"{i:04d}/p_*_*.pt")))
        if len(c_names) == 0 or len(v_names) == 0 or len(p_names) == 0:
            print("Error: No training data found on iteration", i)
            exit(1)
        if not len(c_names) == len(v_names) == len(p_names):
            print("Error: Dataset size mismatch on iteration", i)
            exit(1)
        for j in range(len(c_names)):
            c = torch.load(c_names[j])
            v = torch.load(v_names[j])
            p = torch.load(p_names[j])
            datasets.append(TensorDataset(c, v, p))
            total_size += len(c)
    dataset = ConcatDataset(datasets)
    dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    print("Training data loaded.")

    iteration += 1

    print("Start training...")
    total_train_steps = int(
        math.ceil(total_size / TRAIN_BATCH_SIZE * TRAIN_SAMPLE_RATE)
    )
    nn.train(dataloader, total_train_steps)
    print("Done training.")

    print("Saving checkpoint...")
    nn.save_checkpoint(checkpoint_filepath())
    print("Checkpoint saved.")
