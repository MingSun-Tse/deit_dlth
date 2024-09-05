# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from augment import new_data_aug_generator

import models
import models_v2

import utils
from utils import get_n_params_, get_n_flops_, Timer, strdict_to_dict, strlist_to_list

import warnings
import os
from logger import Logger
pjoin = os.path.join
from pdb import set_trace as st
from pruner import pruner_dict
import wandb

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--print_interval', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=2000)
    parser.add_argument('--plot_interval', type=int, default=100000000)
    parser.add_argument('--block_loss_grad', action="store_true", help="block the grad from loss, only apply weight decay")
    parser.add_argument('--save_interval', type=int, default=2000, help="the interval to save model")
    parser.add_argument('--stage_pr', type=str, default="", help='to appoint layer-wise pruning ratio')
    parser.add_argument('--wg', type=str, default="filter", choices=['filter', 'channel', 'weight'])
    parser.add_argument('--method', type=str, default="", choices=['', 'L1', 'L1_Iter', 'RST', 'RST_Iter'], 
            help='pruning method name; default is "", implying the original training without any pruning')
    parser.add_argument('--save_init_model', action="store_true", help='save the model after initialization')
    parser.add_argument('--base_pr_model', type=str, default=None, help='the model that provides layer-wise pr')
    parser.add_argument('--inherit_pruned', type=str, default='index', choices=['index', 'pr'], 
        help='when --base_pr_model is provided, we can choose to inherit the pruned index or only the pruning ratio (pr)')
    parser.add_argument('--index_layer', type=str, default="numbers", choices=['numbers', 'name_matching'],
            help='the rule to index layers in a network by its name; used in designating pruning ratio')
    parser.add_argument('--pick_pruned', type=str, default='min', choices=['min', 'max', 'rand', 'iter_rand'], help='the criterion to select weights to prune')
    parser.add_argument('--skip_layers', type=str, default="", help='layer id to skip when pruning')
    parser.add_argument('--resume_path', type=str, default=None, help="supposed to replace the original 'resume' feature")
    parser.add_argument('--directly_ft_weights', type=str, default=None, help="the path to a pretrained model")
    parser.add_argument('--base_model_path', type=str, default=None, help="the path to the unpruned base model")
    parser.add_argument('--project_name', type=str, default="")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--screen_print', action="store_true")
    parser.add_argument('--test_pretrained', action="store_true", help='test the pretrained model')
    parser.add_argument('--verbose', action="store_true", help='if true, logprint debug logs')

    # GReg method related (default setting is for ImageNet):
    parser.add_argument('--batch_size_prune', type=int, default=64)
    parser.add_argument('--update_reg_interval', type=int, default=5)
    parser.add_argument('--stabilize_reg_interval', type=int, default=40000)
    parser.add_argument('--lr_prune', type=float, default=1e-4)
    parser.add_argument('--reg_upper_limit', type=float, default=1.0)
    parser.add_argument('--reg_upper_limit_pick', type=float, default=1e-2)
    parser.add_argument('--reg_granularity_pick', type=float, default=1e-5)
    parser.add_argument('--reg_granularity_prune', type=float, default=1e-4)
    parser.add_argument('--reg_granularity_recover', type=float, default=-1e-4)
    parser.add_argument('--RST_schedule', type=str, default='x', choices=['x', 'x^2', 'x^3'])

    # Iterative RST method related
    # parser.add_argument('--batch_size_prune_mini', type=int, default=64)
    # parser.add_argument('--update_reg_interval_mini', type=int, default=1)
    # parser.add_argument('--stabilize_reg_interval_mini', type=int, default=1000)
    # parser.add_argument('--lr_prune_mini', type=float, default=0.001)
    # parser.add_argument('--reg_upper_limit_mini', type=float, default=0.0001)
    # parser.add_argument('--reg_upper_limit_pick_mini', type=float, default=1e-2)
    # parser.add_argument('--reg_granularity_pick_mini', type=float, default=1e-5)
    # parser.add_argument('--reg_granularity_prune_mini', type=float, default=1e-4)
    # parser.add_argument('--reg_granularity_recover_mini', type=float, default=-1e-4)
    # parser.add_argument('--RST_Iter_ft', type = int, default = 0)
    # parser.add_argument('--RST_Iter_weight_delete', action='store_true',
    #         help='if delete the Greged weight in each cycle')

    # LTH related
    parser.add_argument('--num_cycles', type=int, default=0, 
            help='num of cycles in iterative pruning')
    parser.add_argument('--lr_ft_mini', type=str, default='', 
            help='finetuning lr in each iterative pruning cycle')
    parser.add_argument('--epochs_mini', type=int, default=0,
            help='num of epochs in each iterative pruning cycle')
    parser.add_argument('--LTH_Iter', action='store_true',
            help='if use iterative way to make LTH, 0 for no, 1 for yes')
    return parser

parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

logger = Logger(args)
logprint = logger.log_printer.logprint
accprint = logger.log_printer.accprint
netprint = logger.netprint
timer = Timer(args.epochs)

# @mst: use our own save model function
def save_model(state, is_best=False, mark=''):
    out = pjoin(logger.weights_path, "checkpoint.pth")
    torch.save(state, out)
    if is_best:
        out_best = pjoin(logger.weights_path, "checkpoint_best.pth")
        torch.save(state, out_best)
    if mark:
        out_mark = pjoin(logger.weights_path, "checkpoint_{}.pth".format(mark))
        torch.save(state, out_mark)

# @mst: zero out pruned weights for unstructured pruning
def apply_mask_forward(model):
    global mask
    for name, m in model.named_modules():
        if name in mask:
            mask[name] = mask[name].to(m.weight.device)
            m.weight.data.mul_(mask[name])

def main(args):
    wandb.init(
        project=f"{args.project_name}",
        config=args,
        name=args.project_name,
        resume="allow",
        settings=wandb.Settings(console="wrap"),
    )
    wandb.config.update(args)
    
    utils.init_distributed_mode(args)
    logprint(args)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    if args.seed is not None:
        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        # random.seed(seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    logprint(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )

    # @mst: save the model after initialization if necessary
    if args.save_init_model:
        state = {
                'arch': args.model,
                'model': model,
                'state_dict': model.state_dict(),
                'ExpID': logger.ExpID,
        }
        save_model(state, mark='init')

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
        
    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')
            
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    # @mst: load the unpruned model for pruning 
    # This may be useful for the non-imagenet cases where we use our pretrained models
    if args.base_model_path:
        ckpt = torch.load(args.base_model_path)
        if 'state_dict' in ckpt.keys():
            model_without_ddp.load_state_dict(ckpt['state_dict'])
        else:
            model_without_ddp.load_state_dict(ckpt['model'])
        logstr = f"==> Load pretrained model successfully: '{args.base_model_path}'"
        if args.test_pretrained:
            test_stats = evaluate(data_loader_val, model, device)
            acc1, acc5, loss_test = test_stats['acc1'], test_stats['acc5'], test_stats['loss']
            logstr += f". Its accuracy: {acc1:.4f}"
        logprint(logstr)
    
    # @mst: print base model arch
    netprint(model, comment='base model arch')
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = logger.weights_path
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return
    
    # --- @mst: Structured pruning is basically equivalent to providing a new weight initialization before finetune,
    # so just before training, conduct pruning to obtain a new model.
    if args.method:
        # parse for layer-wise prune ratio
        # stage_pr is a list of float, skip_layers is a list of strings
        if args.stage_pr:
            if args.index_layer == 'numbers': # deprecated, kept for now for back-compatability, will be removed
                args.stage_pr = strlist_to_list(args.stage_pr, float) # example: [0, 0.4, 0.5, 0]
                args.skip_layers = strlist_to_list(args.skip_layers, str) # example: [2.3.1, 3.1]
            elif args.index_layer == 'name_matching':
                args.stage_pr = strdict_to_dict(args.stage_pr, float)
        else:
            assert args.base_pr_model, 'If stage_pr is not provided, base_pr_model must be provided'

        # imagenet training costs too much time, so we use a smaller batch size for pruning training
        train_loader_prune = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size_prune, shuffle=(sampler_train is None),
            num_workers=args.num_workers, pin_memory=True, sampler=sampler_train)

        # get the original unpruned model statistics
        n_params_original_v2 = get_n_params_(model)
        n_flops_original_v2 = get_n_flops_(model, img_size=args.input_size, n_channel=3)

        # init some variables for pruning
        prune_state, pruner = 'prune', None
        if args.wg == 'weight':
            global mask

        # resume a model
        if args.resume_path:
            state = torch.load(args.resume_path)
            prune_state = state['prune_state'] # finetune or update_reg or stabilize_reg
            if prune_state == 'finetune':
                model_without_ddp.load_state_dict(state['state_dict'])
                optimizer = create_optimizer(args, model_without_ddp)
                optimizer.load_state_dict(state['optimizer'])
                args.start_epoch = state['epoch']
                logprint("==> Resume model successfully: '{}'. Epoch = {}. prune_state = '{}'".format(
                        args.resume_path, args.start_epoch, prune_state))
            else:
                raise NotImplementedError

        # finetune a model
        if args.directly_ft_weights:
            state = torch.load(args.directly_ft_weights)
            model_without_ddp.load_state_dict(state['state_dict'])
            prune_state = 'finetune'
            logprint("==> Load model successfully: '{}'. Epoch = {}. prune_state = '{}'".format(
                    args.directly_ft_weights, args.start_epoch, prune_state))

            if 'mask' in state:
                mask = state['mask']
                apply_mask_forward(model)
                logprint('==> Mask restored')
                st()

        if prune_state in ['prune']:

            class passer: pass # to pass arguments
            passer.test = evaluate
            passer.finetune = train
            passer.train_loader = train_loader_prune
            passer.test_loader = data_loader_val
            passer.save = save_model
            passer.criterion = criterion
            passer.train_sampler = sampler_train
            passer.pruner = pruner
            passer.args = args
            passer.is_single_branch = is_single_branch
            passer.device = device
            passer.mixup_fn = mixup_fn
            # ***********************************************************
            # key pruning function
            pruner = pruner_dict[args.method].Pruner(model, args, logger, passer)
            model = pruner.prune() # get the pruned model
            if args.method in ['RST', 'RST_Iter']:
                model_before_removing_weights, model = model
                # test the just pruned model
                test_stats = evaluate(data_loader_val, model_before_removing_weights, device) # test set
                acc1, acc5, loss_test = test_stats['acc1'], test_stats['acc5'], test_stats['loss']
                logstr = []
                logstr += ["Acc1 %.4f Acc5 %.4f Loss_test %.4f" % (acc1, acc5, loss_test)]
                accprint(' | '.join(logstr))
            if args.wg == 'weight':
                mask = pruner.mask
                apply_mask_forward(model)
                logprint('==> Check: if the mask does the correct sparsity')
                keys_list = [i for i in mask.keys()]
                pr_list = [ 1-(mask[i].sum()/mask[i].numel()) for i in keys_list ]
                logprint("==> Layer-wise sparsity:")
                logprint(pr_list)
                logprint('==> Apply masks before finetuning to ensure the pruned weights are zero')
            netprint(model, comment='model that was just pruned')
            # ***********************************************************

            # get model statistics of the pruned model
            n_params_now_v2 = get_n_params_(model)
            n_flops_now_v2 = get_n_flops_(model, img_size=args.input_size, n_channel=3)
            logprint("==> n_params_original_v2: {:>9.6f}M, n_flops_original_v2: {:>9.6f}G".format(n_params_original_v2/1e6, n_flops_original_v2/1e9))
            logprint("==> n_params_now_v2:      {:>9.6f}M, n_flops_now_v2:      {:>9.6f}G".format(n_params_now_v2/1e6, n_flops_now_v2/1e9))
            ratio_param = (n_params_original_v2 - n_params_now_v2) / n_params_original_v2
            ratio_flops = (n_flops_original_v2 - n_flops_now_v2) / n_flops_original_v2
            compression_ratio = 1.0 / (1 - ratio_param)
            speedup_ratio = 1.0 / (1 - ratio_flops)
            logprint("==> reduction ratio -- params: {:>5.2f}% (compression ratio {:>.2f}x), flops: {:>5.2f}% (speedup ratio {:>.2f}x)".format(ratio_param*100, compression_ratio, ratio_flops*100, speedup_ratio))
        
            # test the just pruned model
            t1 = time.time()
            test_stats = evaluate(data_loader_val, model, device) # test set
            acc1, acc5, loss_test = test_stats['acc1'], test_stats['acc5'], test_stats['loss']
            logstr = []
            logstr += ["Acc1 %.4f Acc5 %.4f Loss_test %.4f" % (acc1, acc5, loss_test)]
            logstr += ["(test_time %.2fs) Just got pruned model, about to finetune" %  (time.time() - t1)]
            accprint(' | '.join(logstr))
            
            # save the just pruned model
            state = {'arch': args.model,
                    'model': model_without_ddp,
                    'state_dict': model_without_ddp.state_dict(),
                    'acc1': acc1,
                    'acc5': acc5,
                    'ExpID': logger.ExpID,
                    'pruned_wg': pruner.pruned_wg,
                    'kept_wg': pruner.kept_wg,
            }
            if args.wg == 'weight':
                state['mask'] = mask
            save_model(state, mark="just_finished_prune")
    
    # finetune(model, data_loader_train, data_loader_val, sampler_train, criterion, pruner, n_parameters, device, args, dataset_val, loss_scaler, model_ema, mixup_fn, output_dir)
    train(model, criterion, optimizer, lr_scheduler, n_parameters, device, data_loader_train, data_loader_val, dataset_val, loss_scaler, model_ema, mixup_fn, output_dir)

# # @mst
# def finetune(model, data_loader_train, data_loader_val, train_sampler, criterion, pruner, n_parameters, device, args, dataset_val, loss_scaler, model_ema, mixup_fn, output_dir, print_log=True):
#     best_acc1, best_acc1_epoch = 0, 0
#     # since model is new, we need a new optimizer
#     model_without_ddp = model.module
#     optimizer = create_optimizer(args, model_without_ddp)
#     # set lr finetune schduler for finetune
#     lr_scheduler, _ = create_scheduler(args, optimizer)
    
#     acc1_list, loss_train_list, loss_test_list = [], [], []
#     for epoch in range(args.start_epoch, args.epochs):
#         if args.distributed:
#             train_sampler.set_epoch(epoch)
        
#         # @mst: use our own lr scheduler
#         lr = lr_scheduler(optimizer, epoch) if args.method else adjust_learning_rate(optimizer, epoch, args)
#         if print_log:
#             logprint("==> Set lr = %s @ Epoch %d " % (lr, epoch))

#         # train for one epoch
#         train(model, criterion, optimizer, lr_scheduler, n_parameters, device, data_loader_train, data_loader_val, dataset_val, loss_scaler, model_ema, mixup_fn, output_dir)

#         # @mst: check weights magnitude during finetune
#         if args.method in ['GReg-1', 'GReg-2'] and not isinstance(pruner, type(None)):
#             for name, m in model.named_modules():
#                 if name in pruner.reg:
#                     ix = pruner.layers[name].layer_index
#                     mag_now = m.weight.data.abs().mean()
#                     mag_old = pruner.original_w_mag[name]
#                     ratio = mag_now / mag_old
#                     tmp = '[%2d] %25s -- mag_old = %.4f, mag_now = %.4f (%.2f)' % (ix, name, mag_old, mag_now, ratio)
#                     print(tmp, file=logger.logtxt, flush=True)
#                     if args.screen_print:
#                         print(tmp)

#         # evaluate on validation set
#         test_stats = evaluate(data_loader_val, model, device) # train set
#         acc1, acc5, loss_test = test_stats['acc1'], test_stats['acc5'], test_stats['loss']
#         if args.dataset != 'imagenet': # too costly, not test for now
#             train_stats = evaluate(data_loader_train, model, device) # train set
#             acc1_train, acc5_train, loss_train = train_stats['acc1'], train_stats['acc5'], train_stats['loss']
#         else:
#             acc1_train, acc5_train, loss_train = -1, -1, -1
#         acc1_list.append(acc1)
#         loss_train_list.append(loss_train)
#         loss_test_list.append(loss_test)

#         # remember best acc@1 and save checkpoint
#         is_best = acc1 > best_acc1
#         best_acc1 = max(acc1, best_acc1)
#         if is_best:
#             best_acc1_epoch = epoch
#             best_loss_train = loss_train
#             best_loss_test = loss_test
#         if print_log:
#             accprint("Acc1 %.4f Acc5 %.4f Loss_test %.4f | Acc1_train %.4f Acc5_train %.4f Loss_train %.4f | Epoch %d (Best_Acc1 %.4f @ Best_Acc1_Epoch %d) lr %s" % 
#                 (acc1, acc5, loss_test, acc1_train, acc5_train, loss_train, epoch, best_acc1, best_acc1_epoch, lr))
#             logprint('predicted finish time: %s' % timer())

#         ngpus_per_node = torch.cuda.device_count()
#         if not args.multiprocessing_distributed or (args.multiprocessing_distributed
#                 and args.rank % ngpus_per_node == 0):
#             if args.method:
#                 # @mst: use our own save func
#                 state = {'epoch': epoch + 1,
#                         'arch': args.model,
#                         'model': model,
#                         'state_dict': model.state_dict(),
#                         'acc1': acc1,
#                         'acc5': acc5,
#                         'optimizer': optimizer.state_dict(),
#                         'ExpID': logger.ExpID,
#                         'prune_state': 'finetune',
#                 }
#                 if args.wg == 'weight':
#                     state['mask'] = mask 
#                 save_model(state, is_best)
#             else:
#                 save_checkpoint({
#                     'epoch': epoch + 1,
#                     'arch': args.model,
#                     'state_dict': model.state_dict(),
#                     'best_acc1': best_acc1,
#                     'optimizer' : optimizer.state_dict(),
#                 }, is_best)
    
#     last5_acc_mean, last5_acc_std = np.mean(acc1_list[-args.last_n_epoch:]), np.std(acc1_list[-args.last_n_epoch:])
#     last5_loss_train_mean, last5_loss_train_std = np.mean(loss_train_list[-args.last_n_epoch:]), np.std(loss_train_list[-args.last_n_epoch:])
#     last5_loss_test_mean, last5_loss_test_std = np.mean(loss_test_list[-args.last_n_epoch:]), np.std(loss_test_list[-args.last_n_epoch:])
     
#     best = [best_acc1, best_loss_train, best_loss_test]
#     last5 = [last5_acc_mean, last5_acc_std, last5_loss_train_mean, last5_loss_train_std, last5_loss_test_mean, last5_loss_test_std]
#     return best, last5

def train(model, criterion, optimizer, lr_scheduler, n_parameters, device, data_loader_train, data_loader_val, dataset_val, loss_scaler, model_ema, mixup_fn, output_dir):
    model_without_ddp = model.module
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args = args,
            apply_mask_forward = apply_mask_forward,
        )

        lr_scheduler.step(epoch)
        state = {'model': model_without_ddp,
                'state_dict': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'model_ema': get_state_dict(model_ema),
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
        save_model(state)

        test_stats = evaluate(data_loader_val, model, device)

        logprint(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            state = {
                'model': model_without_ddp,
                'state_dict': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'model_ema': get_state_dict(model_ema),
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            save_model(state, is_best=True)
            
        logprint(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        logprint(log_stats)
        wandb.log(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logprint('Training time {}'.format(total_time_str))
    wandb.log({'Training time': total_time_str})

    wandb.finish()

# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')

def is_single_branch(model_name):
    single_branch_model = [
        'mlp_7',
        'vgg',
    ]

    for k in single_branch_model:
        if model_name.startswith(k):
            return True
    return False

if __name__ == '__main__':
    main(args)
