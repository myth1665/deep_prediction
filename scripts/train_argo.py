import argparse
import gc
import logging
import os
import sys
import time

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from sgan.data.loader import data_loader
from sgan.losses_argo import gan_g_loss, gan_d_loss, l2_loss
from sgan.losses_argo import displacement_error, final_displacement_error

from sgan.models_argo import TrajectoryGenerator, TrajectoryDiscriminator
from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path

from sgan.data.data import Argoverse_Social_Data, collate_traj_social
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
torch.backends.cudnn.benchmark = True

import os
from datetime import datetime


# Creating directory 
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y::%H:%M:%S")

dataset_name = 'argoverse'
model_name = 'sgan'

wr_dir = '../runs/' + dataset_name + '/' + model_name + '/' + dt_string + '/'
model_save_dir = '../saved_model/'+ dt_string + '/'
os.makedirs(wr_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

# Writer for tensorboard
writer = SummaryWriter(wr_dir)

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='zara1', type=str)
parser.add_argument('--delim', default=' ')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=200, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=1024, type=int)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=64, type=int)
parser.add_argument('--decoder_h_dim_g', default=128, type=int)
parser.add_argument('--noise_dim', default=None, type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='ped')
parser.add_argument('--clipping_threshold_g', default=0, type=float)
parser.add_argument('--g_learning_rate', default=5e-4, type=float)
parser.add_argument('--g_steps', default=1, type=int)

# Pooling Options
parser.add_argument('--pooling_type', default='pool_net')
parser.add_argument('--pool_every_timestep', default=1, type=bool_flag)

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=1024, type=int)

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str)
parser.add_argument('--encoder_h_dim_d', default=64, type=int)
parser.add_argument('--d_learning_rate', default=5e-4, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Loss Options
parser.add_argument('--l2_loss_weight', default=0, type=float)
parser.add_argument('--best_k', default=1, type=int)

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=5, type=int)
parser.add_argument('--checkpoint_every', default=100, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    long_dtype, float_dtype = get_dtypes(args)

    argoverse_train = Argoverse_Social_Data('../../deep_prediction/data/train/data/')
    argoverse_val   = Argoverse_Social_Data('../../deep_prediction/data/val/data')
    argoverse_test  = Argoverse_Social_Data('../../deep_prediction/data/test_obs/data')

    train_loader = DataLoader(argoverse_train, batch_size=args.batch_size,
                    shuffle=True, num_workers=2,collate_fn=collate_traj_social)
    val_loader = DataLoader(argoverse_val, batch_size=args.batch_size,
                    shuffle=True, num_workers=2,collate_fn=collate_traj_social)
    test_loader = DataLoader(argoverse_test, batch_size=args.batch_size,
                    shuffle=True, num_workers=2,collate_fn=collate_traj_social)

    iterations_per_epoch = len(argoverse_train) / args.batch_size / args.d_steps
    
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)

    generator.apply(init_weights)
    generator.type(float_dtype).train()
    logger.info('Here is the generator:')
    logger.info(generator)

    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type)

    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()
    logger.info('Here is the discriminator:')
    logger.info(discriminator)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=args.d_learning_rate
    )

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        generator.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        optimizer_d.load_state_dict(checkpoint['d_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'G_losses': defaultdict(list),
            'D_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_g': [],
            'norm_d': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'g_state': None,
            'g_optim_state': None,
            'd_state': None,
            'd_optim_state': None,
            'g_best_state': None,
            'd_best_state': None,
            'best_t': None,
            'g_best_nl_state': None,
            'd_best_state_nl': None,
            'best_t_nl': None,
        }
    t0 = None
    
    writer_iter = 0
    
    while t < args.num_iterations:
        gc.collect()
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        
        for batch in train_loader:
            
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.
            if d_steps_left > 0:
                step_type = 'd'
                
                losses_d = discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d)
                
                checkpoint['norm_d'].append( get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
                
            elif g_steps_left > 0:
                step_type = 'g'
                losses_g = generator_step(args, batch, generator,discriminator, g_loss_fn, optimizer_g)
                checkpoint['norm_g'].append( get_total_norm(generator.parameters()))
                g_steps_left -= 1

            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('{} step took {}'.format(step_type, t2 - t1))

            # Skip the rest if we are not at the end of an iteration
            if d_steps_left > 0 or g_steps_left > 0:
                continue

            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(t - 1, time.time() - t0))
                t0 = time.time()

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
                for k, v in sorted(losses_d.items()):
                    logger.info('  [D] {}: {:.3f}'.format(k, v))
                    checkpoint['D_losses'][k].append(v)
                for k, v in sorted(losses_g.items()):
                    logger.info('  [G] {}: {:.3f}'.format(k, v))
                    checkpoint['G_losses'][k].append(v)
                checkpoint['losses_ts'].append(t)
                
                writer.add_scalar('D_data_loss', losses_d['D_data_loss'], t)
                writer.add_scalar('D_total_loss', losses_d['D_total_loss'], t)
                writer.add_scalar('G_discriminator_loss', losses_g['G_discriminator_loss'], t)
                writer.add_scalar('G_l2_loss_rel', losses_g['G_l2_loss_rel'], t)
                writer.add_scalar('G_total_loss', losses_g['G_total_loss'], t)
            
            ### save: D_losses, G_losses
            
            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(args, val_loader, generator, discriminator, d_loss_fn, limit=True)
                
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(args, train_loader, generator, discriminator, d_loss_fn, limit=True)

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                    
                writer.add_scalar('val_ade', metrics_val['ade'], t)
                writer.add_scalar('val_ade_l', metrics_val['ade_l'], t)
                writer.add_scalar('val_ade_nl', metrics_val['ade_nl'], t)
                writer.add_scalar('val_fde', metrics_val['fde'], t)
                writer.add_scalar('val_fde_l', metrics_val['fde_l'], t)
                writer.add_scalar('val_fde_nl', metrics_val['fde_nl'], t)
                
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)
                
                writer.add_scalar('train_ade', metrics_train['ade'], t)
                writer.add_scalar('train_ade_l', metrics_train['ade_l'], t)
                writer.add_scalar('train_ade_nl', metrics_train['ade_nl'], t)
                writer.add_scalar('train_fde', metrics_train['fde'], t)
                writer.add_scalar('train_fde_l', metrics_train['fde_l'], t)
                writer.add_scalar('train_fde_nl', metrics_train['fde_nl'], t)
                
                min_ade = min(checkpoint['metrics_val']['ade'])
                min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])

                if metrics_val['ade'] == min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    checkpoint['g_best_state'] = generator.state_dict()
                    checkpoint['d_best_state'] = discriminator.state_dict()

                if metrics_val['ade_nl'] == min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint['best_t_nl'] = t
                    checkpoint['g_best_nl_state'] = generator.state_dict()
                    checkpoint['d_best_nl_state'] = discriminator.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                checkpoint_path = os.path.join(
                    model_save_dir, '%s_with_model.pt' % args.checkpoint_name
                )
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

                # Save a checkpoint with no model weights by making a shallow
                # copy of the checkpoint excluding some items
                checkpoint_path = os.path.join(
                    model_save_dir, '%s_no_model.pt' % args.checkpoint_name)
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                key_blacklist = [
                    'g_state', 'd_state', 'g_best_state', 'g_best_nl_state',
                    'g_optim_state', 'd_optim_state', 'd_best_state',
                    'd_best_nl_state'
                ]
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)
                logger.info('Done.')

            t += 1
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            
            #### Writer entries here ####
#             writer.add_scalar('Loss/train', np.random.random(), n_iter)
            
            
            
            
            
            
            
            if t >= args.num_iterations:
                writer.close()
                break


def discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d):

    train_agent = batch['train_agent']
    gt_agent = batch['gt_agent']
    neighbour = batch['neighbour']
    neighbour_gt = batch['neighbour_gt']
    av = batch['av']
    av_gt = batch['av_gt']
    seq_path = batch['seq_path']
    seq_id = batch['indexes']
    Rs = batch['rotation']
    ts = batch['translation']

    obs_traj = train_agent[0].unsqueeze(0)
    obs_traj = torch.cat((obs_traj, av[0].unsqueeze(0)),0)
    obs_traj = torch.cat((obs_traj, neighbour[0]),0)

    pred_traj_gt = gt_agent[0].unsqueeze(0)
    pred_traj_gt = torch.cat((pred_traj_gt, av_gt[0].unsqueeze(0)),0)
    pred_traj_gt = torch.cat((pred_traj_gt, neighbour_gt[0]),0)

    ped_count = obs_traj.shape[0]
    seq_start_end = [[0, ped_count]] # last number excluded

    non_linear_ped = []
    _non_linear_ped = [poly_fit(np.array(gt_agent[0]))]
    _non_linear_ped.append(poly_fit(np.array(av_gt[0])))

    for j in range(ped_count-2):
        _non_linear_ped.append(poly_fit(np.array(neighbour_gt[0][j])))
    non_linear_ped += _non_linear_ped

    for i in range(1, len(neighbour)):
        obs_traj = torch.cat((obs_traj, train_agent[i].unsqueeze(0)), 0)
        obs_traj = torch.cat((obs_traj, av[i].unsqueeze(0)),0)
        obs_traj = torch.cat((obs_traj, neighbour[i]), 0)

        pred_traj_gt = torch.cat((pred_traj_gt, gt_agent[i].unsqueeze(0)), 0)
        pred_traj_gt = torch.cat((pred_traj_gt, av_gt[i].unsqueeze(0)),0)
        pred_traj_gt = torch.cat((pred_traj_gt, neighbour_gt[i]), 0)

        seq_start_end.append([ped_count, obs_traj.shape[0]])

        num_peds_considered = obs_traj.shape[0] - ped_count
        ped_count = obs_traj.shape[0]

        _non_linear_ped = [poly_fit(np.array(gt_agent[i]))]
        _non_linear_ped.append(poly_fit(np.array(av_gt[i])))

        for j in range(num_peds_considered-2):
            _non_linear_ped.append(poly_fit(np.array(neighbour_gt[i][j])))

        non_linear_ped += _non_linear_ped

    obs_traj_rel = torch.zeros(obs_traj.shape)
    obs_traj_rel[:,1:,:] = obs_traj[:,1:,:] -  obs_traj[:,:-1,:]    

    pred_traj_gt_rel = torch.zeros(pred_traj_gt.shape)
    pred_traj_gt_rel[:,1:,:] = pred_traj_gt[:,1:,:] - pred_traj_gt[:,0:-1,:]

    seq_start_end = torch.tensor(seq_start_end)
    non_linear_ped = torch.tensor(non_linear_ped).cuda()

    ## 
    obs_traj = obs_traj.transpose_(0,1).cuda() 
    obs_traj_rel = obs_traj_rel.transpose_(0,1).cuda() 
    pred_traj_gt = pred_traj_gt.transpose_(0,1).cuda() 
    pred_traj_gt_rel = pred_traj_gt_rel.transpose_(0,1).cuda() 
    
#     print("Discriminator Batch Loaded !!")
    ###################################################
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)
    
#     print("scores_fake: ", scores_fake)
#     print("scores_real: ", scores_real)
    
    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(),
                                 args.clipping_threshold_d)
    optimizer_d.step()

    return losses


def generator_step(args, batch, generator, discriminator, g_loss_fn, optimizer_g):
    
    train_agent = batch['train_agent']
    gt_agent = batch['gt_agent']
    neighbour = batch['neighbour']
    neighbour_gt = batch['neighbour_gt']
    av = batch['av']
    av_gt = batch['av_gt']
    seq_path = batch['seq_path']
    seq_id = batch['indexes']
    Rs = batch['rotation']
    ts = batch['translation']

    obs_traj = train_agent[0].unsqueeze(0)
    obs_traj = torch.cat((obs_traj, av[0].unsqueeze(0)),0)
    obs_traj = torch.cat((obs_traj, neighbour[0]),0)

    pred_traj_gt = gt_agent[0].unsqueeze(0)
    pred_traj_gt = torch.cat((pred_traj_gt, av_gt[0].unsqueeze(0)),0)
    pred_traj_gt = torch.cat((pred_traj_gt, neighbour_gt[0]),0)

    ped_count = obs_traj.shape[0]
    seq_start_end = [[0, ped_count]] # last number excluded

    non_linear_ped = []
    _non_linear_ped = [poly_fit(np.array(gt_agent[0]))]
    _non_linear_ped.append(poly_fit(np.array(av_gt[0])))

    for j in range(ped_count-2):
        _non_linear_ped.append(poly_fit(np.array(neighbour_gt[0][j])))
    non_linear_ped += _non_linear_ped

    for i in range(1, len(neighbour)):
        obs_traj = torch.cat((obs_traj, train_agent[i].unsqueeze(0)), 0)
        obs_traj = torch.cat((obs_traj, av[i].unsqueeze(0)),0)
        obs_traj = torch.cat((obs_traj, neighbour[i]), 0)

        pred_traj_gt = torch.cat((pred_traj_gt, gt_agent[i].unsqueeze(0)), 0)
        pred_traj_gt = torch.cat((pred_traj_gt, av_gt[i].unsqueeze(0)),0)
        pred_traj_gt = torch.cat((pred_traj_gt, neighbour_gt[i]), 0)

        seq_start_end.append([ped_count, obs_traj.shape[0]])

        num_peds_considered = obs_traj.shape[0] - ped_count
        ped_count = obs_traj.shape[0]

        _non_linear_ped = [poly_fit(np.array(gt_agent[i]))]
        _non_linear_ped.append(poly_fit(np.array(av_gt[i])))

        for j in range(num_peds_considered-2):
            _non_linear_ped.append(poly_fit(np.array(neighbour_gt[i][j])))

        non_linear_ped += _non_linear_ped

    obs_traj_rel = torch.zeros(obs_traj.shape)
    obs_traj_rel[:,1:,:] = obs_traj[:,1:,:] -  obs_traj[:,:-1,:]    

    pred_traj_gt_rel = torch.zeros(pred_traj_gt.shape)
    pred_traj_gt_rel[:,1:,:] = pred_traj_gt[:,1:,:] - pred_traj_gt[:,0:-1,:]

    seq_start_end = torch.tensor(seq_start_end)
    non_linear_ped = torch.tensor(non_linear_ped).cuda()

    ## 
    obs_traj = obs_traj.transpose_(0,1).cuda() 
    obs_traj_rel = obs_traj_rel.transpose_(0,1).cuda() 
    pred_traj_gt = pred_traj_gt.transpose_(0,1).cuda() 
    pred_traj_gt_rel = pred_traj_gt_rel.transpose_(0,1).cuda() 
    
#     print("Generator Batch Loaded !!")
    ##################################################
    
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []
    
    loss_mask = torch.ones(pred_traj_gt.shape[1],30).cuda()

    for _ in range(args.best_k):
        generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
 
        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='raw'))

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    
    
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
#         print("g_l2_loss_rel: ", g_l2_loss_rel.shape)
#         print("seq_start_end: ", seq_start_end)
        
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
#             print("_g_l2_loss_rel __1__ : ", _g_l2_loss_rel)
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
#             print("_g_l2_loss_rel __2__ : ", _g_l2_loss_rel)
#             print("loss mask: ", loss_mask)
#             print("seq_start_end: ", seq_start_end)
#             print("start: ", start, "end: ", end)
#             print("loss mask[start:end]: ", loss_mask[start:end])
            
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(loss_mask[start:end])
#             print("_g_l2_loss_rel __3__ : ", _g_l2_loss_rel)
            g_l2_loss_sum_rel += _g_l2_loss_rel
#             print("g_l2_loss_rel __4__ : ", g_l2_loss_rel)
        

        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), args.clipping_threshold_g
        )
    optimizer_g.step()

    return losses

def poly_fit(traj, traj_len=30, threshold=0.002):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[:, 0], 2, full=True)[1]
    res_y = np.polyfit(t, traj[:, 1], 2, full=True)[1]
    
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
    
def check_accuracy(args, loader, generator, discriminator, d_loss_fn, limit=True):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    generator.eval()
    
    with torch.no_grad():
        for batch in loader:      
            train_agent = batch['train_agent']
            gt_agent = batch['gt_agent']
            neighbour = batch['neighbour']
            neighbour_gt = batch['neighbour_gt']
            av = batch['av']
            av_gt = batch['av_gt']
            seq_path = batch['seq_path']
            seq_id = batch['indexes']
            Rs = batch['rotation']
            ts = batch['translation']

            obs_traj = train_agent[0].unsqueeze(0)
            obs_traj = torch.cat((obs_traj, av[0].unsqueeze(0)),0)
            obs_traj = torch.cat((obs_traj, neighbour[0]),0)

            pred_traj_gt = gt_agent[0].unsqueeze(0)
            pred_traj_gt = torch.cat((pred_traj_gt, av_gt[0].unsqueeze(0)),0)
            pred_traj_gt = torch.cat((pred_traj_gt, neighbour_gt[0]),0)

            ped_count = obs_traj.shape[0]
            seq_start_end = [[0, ped_count]] # last number excluded

            non_linear_ped = []
            _non_linear_ped = [poly_fit(np.array(gt_agent[0]))]
            _non_linear_ped.append(poly_fit(np.array(av_gt[0])))

            for j in range(ped_count-2):
                _non_linear_ped.append(poly_fit(np.array(neighbour_gt[0][j])))
            non_linear_ped += _non_linear_ped

            for i in range(1, len(neighbour)):
                obs_traj = torch.cat((obs_traj, train_agent[i].unsqueeze(0)), 0)
                obs_traj = torch.cat((obs_traj, av[i].unsqueeze(0)),0)
                obs_traj = torch.cat((obs_traj, neighbour[i]), 0)

                pred_traj_gt = torch.cat((pred_traj_gt, gt_agent[i].unsqueeze(0)), 0)
                pred_traj_gt = torch.cat((pred_traj_gt, av_gt[i].unsqueeze(0)),0)
                pred_traj_gt = torch.cat((pred_traj_gt, neighbour_gt[i]), 0)

                seq_start_end.append([ped_count, obs_traj.shape[0]])

                num_peds_considered = obs_traj.shape[0] - ped_count
                ped_count = obs_traj.shape[0]

                _non_linear_ped = [poly_fit(np.array(gt_agent[i]))]
                _non_linear_ped.append(poly_fit(np.array(av_gt[i])))

                for j in range(num_peds_considered-2):
                    _non_linear_ped.append(poly_fit(np.array(neighbour_gt[i][j])))

                non_linear_ped += _non_linear_ped

            obs_traj_rel = torch.zeros(obs_traj.shape)
            obs_traj_rel[:,1:,:] = obs_traj[:,1:,:] -  obs_traj[:,:-1,:]    

            pred_traj_gt_rel = torch.zeros(pred_traj_gt.shape)
            pred_traj_gt_rel[:,1:,:] = pred_traj_gt[:,1:,:] - pred_traj_gt[:,0:-1,:]

            seq_start_end = torch.tensor(seq_start_end)
            non_linear_ped = torch.tensor(non_linear_ped).cuda()

            ## 
            obs_traj = obs_traj.transpose_(0,1).cuda() 
            obs_traj_rel = obs_traj_rel.transpose_(0,1).cuda() 
            pred_traj_gt = pred_traj_gt.transpose_(0,1).cuda() 
            pred_traj_gt_rel = pred_traj_gt_rel.transpose_(0,1).cuda() 
            
            ################################################################
               
            linear_ped = 1 - non_linear_ped
            
            loss_mask = torch.ones(pred_traj_gt.shape[1],30).cuda()
#             loss_mask = loss_mask[:, args.obs_len:].cuda()

            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            g_l2_loss_abs, g_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade, ade_l, ade_nl = cal_ade(pred_traj_gt[:,0,:].unsqueeze(0), pred_traj_fake[:,0,:].unsqueeze(0), linear_ped[0], non_linear_ped[0])

            fde, fde_l, fde_nl = cal_fde(pred_traj_gt[:,0,:].unsqueeze(0), pred_traj_fake[:,0,:].unsqueeze(0), linear_ped[0], non_linear_ped[0])

            traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
            scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            
#             logger.info('limit = {}, total_traj = {}, num_samples_check = {}'.format(limit, total_traj, args.num_samples_check))
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / loss_mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    generator.train()
    return metrics


def cal_l2_losses(pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,loss_mask):
    g_l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, loss_mask, mode='sum')
    g_l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum')
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(
    pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], linear_ped)
    fde_nl = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped)
    return fde, fde_l, fde_nl


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
