import argparse
import os
from tqdm import tqdm
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchdiffeq import odeint

from NeuralMD.datasets.MISATO import DatasetMISATOSemiFlexibleSingleTrajectory
from NeuralMD.dataloaders.dataloader_MISATO import DataLoaderMISATO
from NeuralMD.evaluation import get_matching_list, get_stability_list, get_ligand_collision_list, get_binding_collision_list_semi_flexible
from models.VerletMD_Binding import VerletMD
from torch_ema import ExponentialMovingAverage


def save_model(save_best):
    if not args.output_model_dir == "":
        if save_best:
            print("save model with optimal loss")
            output_model_path = os.path.join(args.output_model_dir, "model.pth")
            saved_model_dict = {}
            saved_model_dict["binding_model"] = binding_model.state_dict()
            torch.save(saved_model_dict, output_model_path)

        else:
            print("save model in the last epoch")
            output_model_path = os.path.join(args.output_model_dir, "model_final.pth")
            saved_model_dict = {}
            saved_model_dict["binding_model"] = binding_model.state_dict()
            torch.save(saved_model_dict, output_model_path)
    return


def train(loader):
    binding_model.train()

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader

    accum_loss, accum_count = 0, 0
    start_time = time.time()

    for batch_idx, batch in enumerate(L):
        batch = batch.to(device)

        start_traj_idx = 0
        end_traj_idx = train_end_split_idx

        for traj_idx in range(start_traj_idx, end_traj_idx):
            pos_input = batch.ligand_trajectory_pos[:, traj_idx, :].clone()
            energy_target = batch.energy[0, traj_idx].clone()

            condition_ligand = (batch.ligand_x, batch.batch_ligand, batch.ligand_mass)
            pos_N = batch.protein_pos[batch.mask_n]
            pos_Ca = batch.protein_pos[batch.mask_ca]
            pos_C = batch.protein_pos[batch.mask_c]
            condition_protein = (pos_N, pos_Ca, pos_C, batch.protein_backbone_residue, batch.batch_residue)
            condition = condition_ligand + condition_protein

            energy_pred = binding_model(pos_input, condition)

            loss = criterion(energy_pred, energy_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()

            accum_loss += loss.item()
            accum_count += 1
    
    if args.lr_scheduler in ["StepLR", "CosineAnnealingLR"]:
        lr_scheduler.step()
    elif args.lr_scheduler in ["ReduceLROnPlateau"]:
        lr_scheduler.step(accum_loss / accum_count)
    
    print("loss: {:.5f}\t\t{:.3f}s".format(accum_loss / accum_count, time.time() - start_time))
    return


def evaluate(loader, return_trajectory=False):
    start_time = time.time()
    binding_model.eval()

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader

    rmse_loss, mae_loss, eval_count = 0, 0, 0
    matching_list, stability_list, ligand_collision_list, binding_collision_list = [], [], [], []

    for batch_idx, batch in enumerate(L):
        batch = batch.to(device)

        start_traj_idx = train_end_split_idx - 1
        pos_input = batch.ligand_trajectory_pos[:, start_traj_idx, :].clone()
        vel_input = batch.ligand_trajectory_pos[:, start_traj_idx+1, :].clone() - batch.ligand_trajectory_pos[:, start_traj_idx, :].clone()
        trajectory_pred = []

        for traj_idx in range(start_traj_idx, 99):
            pos_target = batch.ligand_trajectory_pos[:, traj_idx+1, :]

            condition_ligand = (batch.ligand_x, batch.batch_ligand, batch.ligand_mass)
            pos_N = batch.protein_pos[batch.mask_n]
            pos_Ca = batch.protein_pos[batch.mask_ca]
            pos_C = batch.protein_pos[batch.mask_c]
            condition_protein = (pos_N, pos_Ca, pos_C, batch.protein_backbone_residue, batch.batch_residue)
            condition = condition_ligand + condition_protein
            pos_pred, vel_pred = binding_model.move([pos_input, vel_input], condition)

            trajectory_pred.append(pos_pred.cpu())

            rmse_loss = rmse_loss + torch.sum((pos_pred - pos_target).pow(2).sum(dim=1).sqrt()).cpu().item()
            mae_loss = mae_loss + torch.sum(torch.abs(pos_pred - pos_target)).cpu().item()

            eval_count += pos_pred.shape[0]

            pos_input = pos_pred.clone()
            vel_input = vel_pred.clone()
    
        trajectory_target = batch.ligand_trajectory_pos.cpu()[:, start_traj_idx+1:, :].transpose(0, 1).detach()
        trajectory_pred = torch.stack(trajectory_pred).detach()
        matching_list.extend(get_matching_list(trajectory_target, trajectory_pred, batch=batch.batch_ligand))
        stability_list.extend(get_stability_list(trajectory_target, trajectory_pred, batch=batch.batch_ligand))
        ligand_collision_list.extend(get_ligand_collision_list(trajectory_pred, batch.ligand_x, batch=batch.batch_ligand))
        batch_protein = batch.batch_residue.unsqueeze(0).expand([3, -1]).contiguous().view(-1)
        protein_x = torch.ones((batch_protein.shape[0],))
        protein_x[batch.mask_n] = 6
        protein_x[batch.mask_ca] = 5
        protein_x[batch.mask_c] = 5
        binding_collision_list.extend(get_binding_collision_list_semi_flexible(trajectory_pred, batch.ligand_x, batch.protein_pos.cpu(), protein_x, batch_ligand=batch.batch_ligand, batch_protein=batch_protein))

        ########## NOTE: This is tricky, only for single-trajectory ##########
        ligand_trajectory_target_list, ligand_trajectory_pred_list = trajectory_target.numpy(), trajectory_pred.numpy()
        ligand_atom_list, protein_target_list = batch.ligand_x.cpu().numpy(), batch.protein_pos.cpu().numpy()
        mask_n, mask_ca, mask_c = batch.mask_n.cpu().numpy(), batch.mask_ca.cpu().numpy(), batch.mask_c.cpu().numpy()


    mae_loss = mae_loss / eval_count
    rmse_loss = rmse_loss / eval_count
    matching = np.mean(matching_list)
    stability = np.mean(stability_list)
    ligand_collision = np.mean(ligand_collision_list)
    binding_collision = np.mean(binding_collision_list)
    
    total_frame_count = (100 - train_end_split_idx) * len(matching_list)
    total_time = time.time() - start_time
    FPS = total_frame_count / total_time
    if not return_trajectory:
        return mae_loss, rmse_loss, matching, stability, ligand_collision, binding_collision, FPS
    else:
        return mae_loss, rmse_loss, matching, stability, ligand_collision, binding_collision, FPS, \
            ligand_trajectory_target_list, ligand_trajectory_pred_list, ligand_atom_list, protein_target_list, mask_n, mask_ca, mask_c


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--loss", type=str, default="MAE", choices=["RMSE", "MAE"])
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "Adam"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="None")
    parser.add_argument("--decay", type=float, default=0)
    parser.add_argument("--print_every_epoch", type=int, default=5)
    parser.add_argument("--output_model_dir", type=str, default="")
    
    parser.add_argument("--input_data_dir", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="MISATO")
    parser.add_argument("--PDB_ID", type=str, default="5WIJ")

    parser.add_argument("--eval_train", dest="eval_train", action="store_true")
    parser.add_argument("--no_eval_train", dest="eval_train", action="store_false")
    parser.set_defaults(eval_train=True)

    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--no_verbose", dest="verbose", action="store_false")
    parser.set_defaults(verbose=False)

    parser.add_argument("--model_3d_ligand", type=str, default="FrameNet01")
    parser.add_argument("--model_3d_protein", type=str, default="FrameNetProtein03")

    # for modeling
    parser.add_argument("--emb_dim", type=int, default=128)

    # for FrameNet
    parser.add_argument("--FrameNet_cutoff", type=float, default=5)
    parser.add_argument("--FrameNet_num_layers", type=int, default=4)
    parser.add_argument("--FrameNet_complex_layer", type=int, default=1)
    parser.add_argument("--FrameNet_num_radial", type=int, default=96)
    parser.add_argument("--FrameNet_rbf_type", type=str, default="RBF_repredding_01")
    parser.add_argument("--FrameNet_gamma", type=float, default=None)
    parser.add_argument("--FrameNet_readout", type=str, default="mean")

    args = parser.parse_args()
    print("args", args)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_end_split_idx = 80

    binding_model = VerletMD(args).to(device)

    data_root = os.path.join(args.input_data_dir, args.dataset)
    dataset = DatasetMISATOSemiFlexibleSingleTrajectory(root=data_root, PDB_ID=args.PDB_ID)
    loader = DataLoaderMISATO(dataset, batch_size=1, num_workers=args.num_workers, shuffle=True)

    if args.loss == "RMSE":
        criterion = nn.RMSELoss(reduction="mean")
    elif args.loss == "MAE":
        criterion = nn.L1Loss(reduction="mean")

    # set up optimizer
    model_param_group = [
        {"params": binding_model.parameters(), "lr": args.lr},
    ]
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model_param_group, lr=args.lr, weight_decay=args.decay)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
        
    ema = ExponentialMovingAverage(binding_model.parameters(), decay=0.995)

    lr_scheduler = None
    if args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs
        )
        print("Apply lr scheduler CosineAnnealingLR")
    elif args.lr_scheduler == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor
        )
        print("Apply lr scheduler StepLR")
    elif args.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.lr_decay_factor, patience=args.lr_decay_patience, min_lr=args.min_lr
        )
        print("Apply lr scheduler ReduceLROnPlateau")

    train_mae_list, test_mae_list = [], []
    train_rmse_list, test_rmse_list = [], []
    train_matching_list, test_matching_list = [], []
    train_stability_list, test_stability_list = [], []
    train_ligand_collision_list, test_ligand_collision_list = [], []
    train_binding_collision_list, test_binding_collision_list = [], []
    result_list = []
    best_train_mae, best_train_idx = 1e10, 0

    print("Initial")
    test_mae, test_rmse, test_matching, test_stability, test_ligand_collision, test_binding_collision, FPS = evaluate(loader)
    print("MAE test: {:.5f}".format(test_mae))
    print("RMSE test: {:.5f}".format(test_rmse))
    print("Matching test: {:.5f}".format(test_matching))
    print("Stability test: {:.5f}".format(test_stability))
    print("Ligand collision test: {:.5f}".format(test_ligand_collision))
    print("Ligand collision test: {:.5f}".format(test_binding_collision))
    print()

    for e in range(1, args.epochs+1):
        print("epoch {}".format(e))
        train(loader)

        if e % args.print_every_epoch == 0:
            if args.eval_train:
                train_mae, train_rmse, train_matching, train_stability, train_ligand_collision, train_binding_collision, _ = evaluate(loader)
            else:
                train_mae, train_rmse, train_matching, train_stability, train_ligand_collision, train_binding_collision = 0, 0, 0, 0, 0
            test_mae, test_rmse, test_matching, test_stability, test_ligand_collision, test_binding_collision, _, \
                ligand_trajectory_target_list, ligand_trajectory_pred_list, ligand_atom_list, protein_target_list, mask_n, mask_ca, mask_c = evaluate(loader, return_trajectory=True)

            train_mae_list.append(train_mae)
            train_rmse_list.append(train_rmse)
            train_matching_list.append(train_matching)
            train_stability_list.append(train_stability)
            train_ligand_collision_list.append(train_ligand_collision)
            train_binding_collision_list.append(train_binding_collision)
            test_mae_list.append(test_mae)
            test_rmse_list.append(test_rmse)
            test_matching_list.append(test_matching)
            test_stability_list.append(test_stability)
            test_ligand_collision_list.append(test_ligand_collision)
            test_binding_collision_list.append(test_binding_collision)
            result_list.append((ligand_trajectory_target_list, ligand_trajectory_pred_list, ligand_atom_list, protein_target_list, mask_n, mask_ca, mask_c))

            if train_mae <= best_train_mae:
                best_train_mae = train_mae
                best_train_idx = len(train_mae_list) - 1
                save_model(save_best=True)

            print("MAE train: {:.5f}\t\ttest: {:.5f}".format(train_mae, test_mae))
            print("RMSE train: {:.5f}\t\ttest: {:.5f}".format(train_rmse, test_rmse))
            print("Matching train: {:.5f}\t\ttest: {:.5f}".format(train_matching, test_matching))
            print("Stability train: {:.5f}\t\ttest: {:.5f}".format(train_stability, test_stability))
            print("Ligand collision train: {:.5f}\t\ttest: {:.5f}".format(train_ligand_collision, test_ligand_collision))
            print("Binding collision train: {:.5f}\t\ttest: {:.5f}".format(train_binding_collision, test_binding_collision))
            print()

    print("best MAE train: {:.6f}\ttest: {:.6f}".format(
        train_mae_list[best_train_idx], test_mae_list[best_train_idx],
    ))
    print("best RMSE train: {:.6f}\ttest: {:.6f}".format(
        train_rmse_list[best_train_idx], test_rmse_list[best_train_idx],
    ))
    print("best Matching train: {:.6f}\ttest: {:.6f}".format(
        train_matching_list[best_train_idx], test_matching_list[best_train_idx],
    ))
    print("best Stability train: {:.6f}\ttest: {:.6f}".format(
        train_stability_list[best_train_idx], test_stability_list[best_train_idx],
    ))
    print("best Ligand collision train: {:.6f}\ttest: {:.6f}".format(
        train_ligand_collision_list[best_train_idx], test_ligand_collision_list[best_train_idx],
    ))
    print("best Binding collision train: {:.6f}\ttest: {:.6f}".format(
        train_binding_collision_list[best_train_idx], test_binding_collision_list[best_train_idx],
    ))
    print("FPS: {:.5f}".format(FPS))

    ligand_trajectory_target_list, ligand_trajectory_pred_list, ligand_atom_list, protein_target_list, mask_n, mask_ca, mask_c = result_list[best_train_idx]
    np.savez("{}/trajectory".format(args.output_model_dir),
        ligand_trajectory_target_list=ligand_trajectory_target_list, ligand_trajectory_pred_list=ligand_trajectory_pred_list,
        ligand_atom_list=ligand_atom_list, protein_target_list=protein_target_list,
        mask_n=mask_n, mask_ca=mask_ca, mask_c=mask_c,
    )
    save_model(save_best=False)