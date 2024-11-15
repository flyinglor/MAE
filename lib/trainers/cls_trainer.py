import os
import math
import time
from functools import partial
from matplotlib.pyplot import grid
import numpy as np
from numpy import nanmean, nonzero, percentile
from torchprofile import profile_macs

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import sys
sys.path.append('..')

import models
import networks
from utils import SmoothedValue, concat_all_gather, LayerDecayValueAssigner

import wandb

from lib.data.med_transforms import get_scratch_train_transforms, get_val_transforms, get_post_transforms, get_vis_transforms, get_raw_transforms
from lib.data.med_datasets import get_msd_trainset, get_train_loader, get_val_loader, idx2label_all, btcv_8cls_idx, get_test_loader
from lib.tools.visualization import patches3d_to_grid, images3d_to_grid
from .base_trainer import BaseTrainer

from timm.data import Mixup
from timm.utils import accuracy
from timm.models.layers.helpers import to_3tuple

from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import get_confusion_matrix, compute_confusion_matrix_metric, compute_meandice
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from collections import defaultdict, OrderedDict

import pdb

class ClsTrainer(BaseTrainer):
    r"""
    General Classification Trainer
    """
    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.proj_name
        self.scaler = torch.cuda.amp.GradScaler()

        # self.fivefolds_test_loss = []
        # self.fivefolds_test_accuracy = []
        # self.fivefolds_test_bal_accuracy = []
        # self.fivefold_test_precision = []
        # self.fivefold_test_recall = []
        # self.fivefold_test_f1 = []

        #TODO: should i put cross entropy loss here?
        self.metric_funcs = OrderedDict([
                                        ('Dice', 
                                          compute_meandice)
                                        ])

        # if args.test:
        #     self.metric_funcs = OrderedDict([
        #                                 ('Dice', 
        #                                   compute_meandice),
        #                                 ('HD',
        #                                   partial(compute_hausdorff_distance, percentile=95))
        #                                 ])
        # else:
        #     self.metric_funcs = OrderedDict([
        #                                 ('Dice', 
        #                                   compute_meandice)
        #                                 ])

    def build_model(self):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args
            print(f"=> creating model {self.model_name}")

            # if args.dataset == 'btcv':
            #     args.num_classes = 14
            #     self.loss_fn = DiceCELoss(to_onehot_y=True,
            #                               softmax=True,
            #                               squared_pred=True,
            #                               smooth_nr=args.smooth_nr,
            #                               smooth_dr=args.smooth_dr)
            # elif args.dataset == 'msd_brats':
            #     args.num_classes = 3
            #     self.loss_fn = DiceLoss(to_onehot_y=False, 
            #                             sigmoid=True, 
            #                             squared_pred=True, 
            #                             smooth_nr=args.smooth_nr, 
            #                             smooth_dr=args.smooth_dr)
            if args.dataset in ['hospital','dzne']:
                args.num_classes = 3
                self.loss_fn = DiceCELoss(to_onehot_y=True,
                                          softmax=False,
                                          squared_pred=False)
            else:
                raise ValueError(f"Unsupported dataset {args.dataset}")
            
            #TODO: figure out what this post transform is
            # self.post_pred, self.post_label = get_post_transforms(args)

            # setup mixup and loss functions
            # if args.mixup > 0:
            #     raise NotImplemented("Mixup for segmentation has not been implemented.")
            # else:
            #     self.mixup_fn = None

            self.model = getattr(models, self.model_name)(encoder=getattr(networks, args.enc_arch),
                                                          decoder=getattr(networks, args.dec_arch),
                                                          args=args)

            # load pretrained weights
            # for testing
            if hasattr(args, 'test') and args.test and args.pretrain is not None and os.path.exists(args.pretrain):
                print(f"=> Start loading the model weights from {args.pretrain} for test")
                checkpoint = torch.load(args.pretrain, map_location='cpu')
                state_dict = checkpoint['state_dict']
                msg = self.model.load_state_dict(state_dict, strict=False)
                print(f'Loading messages: \n {msg}')
                print(f"=> Finish loading pretrained weights from {args.pretrain}")
            # for pretraining
            elif args.pretrain is not None and os.path.exists(args.pretrain):
                print(f"=> Start loading pretrained weights from {args.pretrain}")
                checkpoint = torch.load(args.pretrain, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                # import pdb
                # pdb.set_trace()

                #TODO: load state dict, encoder
                if self.model_name == 'MAE3DFINETUNE':
                    #TODO: can i load the whole model like this or i need to just load the encoder?
                    # msg = self.model.load_state_dict(state_dict, strict=False)
                    #delete decoder
                    for key in list(state_dict.keys()):
                        # print(key)
                        if key.startswith('decoder'):
                            del state_dict[key]
                        if key == 'encoder_to_decoder.weight':
                            del state_dict[key]
                        if key == 'encoder_to_decoder.bias':
                            del state_dict[key]
                        if key == 'mask_token':
                            del state_dict[key]
                            #TODO: does mae encoder encoder.pos_embed exist?
                        # if key == 'encoder_pos_embed' and \
                        #     state_dict['encoder_pos_embed'].shape != self.model.encoder.encoder_pos_embed.shape:
                        #     del state_dict[key]
                    
                    # Initialize lists to keep track of mismatches
                    shape_mismatch_keys = []
                    model_state_dict = self.model.state_dict()

                    # Compare shapes
                    for key in state_dict.keys():
                        if key in model_state_dict:
                            if state_dict[key].shape != model_state_dict[key].shape:
                                shape_mismatch_keys.append(key)

                    if shape_mismatch_keys:
                        print("Shape mismatches found for the following keys:")
                        for key in shape_mismatch_keys:
                            print(f"{key}: model shape {model_state_dict[key].shape}, state_dict shape {state_dict[key].shape}")
                    else:
                        print("No shape mismatches found.")

                    msg = self.model.load_state_dict(state_dict, strict=False)

                print(f'Loading messages: \n {msg}')
                print(f"=> Finish loading pretrained weights from {args.pretrain}")

            self.wrap_model()
        elif self.model_name == 'Unknown':
            raise ValueError("=> Model name is still unknown")
        else:
            raise ValueError("=> Model has been created. Do not create twice")

    def build_optimizer(self):
        assert(self.model is not None and self.wrapped_model is not None), \
                "Model is not created and wrapped yet. Please create model first."
        print("=> creating optimizer")
        args = self.args
        model = self.model

        num_layers = model.get_num_layers()
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))

        # optim_params = self.group_params(model)
        #TODO: only finetuning encoder????
        optim_params = self.get_parameter_groups(get_layer_id=partial(assigner.get_layer_id, prefix='encoder.'), 
                                                 get_layer_scale=assigner.get_scale, 
                                                 verbose=True)
        # TODO: create optimizer factory
        self.optimizer = torch.optim.AdamW(optim_params, 
                                            lr=args.lr,
                                            betas=(args.beta1, args.beta2),
                                            weight_decay=args.weight_decay)

        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

    def build_dataloader(self):
        if self.dataloader is None:
            print("=> creating train dataloader")
            args = self.args

            if args.dataset in ['btcv', 'msd_brats', 'adni', 'hospital','dzne']:
                # build train dataloader
                if not args.test:
                    train_transform = get_raw_transforms(args)
                    self.dataloader = get_train_loader(args, 
                                                    batch_size=self.batch_size, 
                                                    workers=self.workers, 
                                                    train_transform=train_transform)
                    self.iters_per_epoch = len(self.dataloader)
                    print(f"==> Length of train dataloader is {self.iters_per_epoch}")
                else:
                    self.dataloader = None
                # build val dataloader
                val_transform = get_val_transforms(args)
                self.val_dataloader = get_val_loader(args, 
                                                     batch_size=args.batch_size, # batch per gpu
                                                     workers=self.workers, 
                                                     val_transform=val_transform)
                print(f"==> Length of val dataloader is {len(self.val_dataloader)}")

                # build test dataloader
                self.test_dataloader = get_test_loader(args, 
                                                       batch_size=args.batch_size, # batch per gpu
                                                       workers=self.workers, 
                                                       val_transform=val_transform)
                print(f"==> Length of test dataloader is {len(self.test_dataloader)}")
                # TODO: don't need vis anymore?
                # build vis dataloader
                # vis_transform = get_vis_transforms(args)
                # self.vis_dataloader = get_val_loader(args,
                #                                     batch_size=args.vis_batch_size,
                #                                     workers=self.workers,
                #                                     val_transform=vis_transform)
            elif args.dataset == 'brats20':
                raise NotImplementedError("brats20 transforms and dataloaders on MONAI has not been implemented yet.")
            else:
                raise ValueError("Currently only support brats2020 dataset")
        else:
            raise ValueError(f"Dataloader has been created. Do not create twice.")
        print("=> finish creating dataloader")

    def run(self):
        # for f in range(1,6):
        # self.args.folds = f
        args = self.args
        # Compute iterations when resuming
        niters = args.start_epoch * self.iters_per_epoch

        best_metric = 100
        best_ts_metric = 100
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                self.dataloader.sampler.set_epoch(epoch)
                torch.distributed.barrier()
            
            if epoch == args.start_epoch:
                self.evaluate(epoch=epoch, niters=niters)

            # train for one epoch
            niters = self.epoch_train(epoch, niters)

            # evaluate after each epoch training
            if (epoch + 1) % args.eval_freq == 0:
                metric = self.evaluate(epoch=epoch, niters=niters, test=False) #loss
                if metric < best_metric:
                    print(f"=> New val best metric: {metric} | Old val best metric: {best_metric}!")
                    best_metric = metric
                    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                        self.save_checkpoint(
                            {
                                'epoch': epoch + 1,
                                'arch': args.arch,
                                'state_dict': self.model.state_dict(),
                                'optimizer' : self.optimizer.state_dict(),
                                'scaler': self.scaler.state_dict(), # additional line compared with base imple
                                'metric':metric
                            }, 
                            is_best=False, 
                            filename=f'{args.ckpt_dir}/best_model.pth.tar'
                        )
                        print("=> Finish saving best model.")
                else:
                    print(f"=> Still old val best metric: {best_metric}")


            # if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
            #     if (epoch + 1) == args.epochs :
            #         #TODO: save the last model
            #         self.save_checkpoint(
            #             {
            #                 'epoch': epoch + 1,
            #                 'arch': args.arch,
            #                 'state_dict': self.model.state_dict(),
            #                 'optimizer' : self.optimizer.state_dict(),
            #                 'scaler': self.scaler.state_dict(), # additional line compared with base imple
            #             }, 
            #             is_best=False, 
            #             filename=f'{args.ckpt_dir}/last_model_fold{args.fold}.pth.tar'
            #         )
        return self.test_metrics()
        
        #after 5 folds, print average
        # print(f"Average Test Loss: {np.mean(self.fivefolds_test_loss)}, std: {np.std(self.fivefolds_test_loss)}")
        # print(f"Average Test Accuracy: {np.mean(self.fivefolds_test_accuracy)}, std: {np.std(self.fivefolds_test_accuracy)}")
        # print(f"Average Test Balanced Accuracy: {np.mean(self.fivefolds_test_bal_accuracy)}, std: {np.std(self.fivefolds_test_bal_accuracy)}")
        # print(f"Average Test Precision: {np.mean(self.fivefold_test_precision)}, std: {np.std(self.fivefold_test_precision)}")
        # print(f"Average Test Recall: {np.mean(self.fivefold_test_recall)}, std: {np.std(self.fivefold_test_recall)}")
        # print(f"Average Test F1: {np.mean(self.fivefold_test_f1)}, std: {np.std(self.fivefold_test_f1)}")


    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        model = self.wrapped_model
        optimizer = self.optimizer
        scaler = self.scaler
        # mixup_fn = self.mixup_fn
        loss_fn = self.loss_fn

        # switch to train mode
        model.train()

        load_start_time = time.time()
        for i, batch_data in enumerate(train_loader):
            load_time = time.time() - load_start_time
            # adjust learning at the beginning of each iteration
            # self.scheduler.step(epoch)
            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)

            if args.dataset in ['hospital','dzne','adni']:
                image = batch_data[0]
                target = batch_data[1]
                # print(image.shape)
            else:
                image = batch_data['image']
                target = batch_data['label']

            # print(image.shape)

            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output and loss
            forward_start_time = time.time()
            # forward_start_time_1 = time.perf_counter()
            with torch.cuda.amp.autocast(True):
                loss = self.train_class_batch(model, image, target, loss_fn)
            forward_time = time.time() - forward_start_time

            # compute gradient and do SGD step
            bp_start_time = time.time()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bp_time = time.time() - bp_start_time

            # torch.cuda.synchronize()
            # print(f"training iter time is {time.perf_counter() - forward_start_time_1}")

            # Log to the screen
            if i % args.print_freq == 0:
                if 'lr_scale' in optimizer.param_groups[0]:
                    last_layer_lr = optimizer.param_groups[0]['lr'] / optimizer.param_groups[0]['lr_scale']
                else:
                    last_layer_lr = optimizer.param_groups[0]['lr']

                print(f"Epoch: {epoch:03d}/{args.epochs} | "
                      f"Iter: {i:05d}/{self.iters_per_epoch} | "
                      f"TotalIter: {niters:06d} | "
                      f"Init Lr: {self.lr:.05f} | "
                      f"Lr: {last_layer_lr:.05f} | "
                      f"Load Time: {load_time:.03f}s | "
                      f"Forward Time: {forward_time:.03f}s | "
                      f"Backward Time: {bp_time:.03f}s | "
                      f"Loss: {loss.item():.03f}")
                if args.rank == 0 and not args.disable_wandb:
                    wandb.log(
                        {
                        f"Fold {args.fold} - lr": last_layer_lr,
                        f"Fold {args.fold} - Training Loss": loss.item(),
                        "custom_step": niters,
                        },
                        # step=niters,
                    )

            niters += 1
            load_start_time = time.time()
        return niters

    @staticmethod
    def train_class_batch(model, samples, target, criterion):
        outputs = F.softmax(model(samples), dim=1)
        #change to cross entropy loss
        loss = criterion.ce(outputs, target)
        return loss

    @torch.no_grad()
    def evaluate(self, epoch=0, niters=0, test=False):
        print("=> Start Evaluating")
        args = self.args
        model = self.wrapped_model
        val_loader = self.val_dataloader
        val_samples = len(val_loader)
        print(f"val samples: {val_samples}")

        # switch to evaluation mode
        model.eval()
        valid_losses = []
        for i, batch_data in enumerate(val_loader):
            if args.dataset in ['hospital','dzne', 'adni']:
                image = batch_data[0]
                target = batch_data[1]
                # print(image.shape)
            else:
                image = batch_data['image']
                target = batch_data['label']
            # image, target = batch_data['image'], batch_data['label']
            if args.gpu is not None:
                image = image.to(args.gpu, non_blocking=True)
                target = target.to(args.gpu, non_blocking=True)

            # get the validation loss
            outputs = F.softmax(model(image), dim=1)
            loss = self.loss_fn.ce(outputs, target)
            #TODO: need .item()?
            valid_losses.append(loss.item())

            print(f'==> Evaluating on the {i+1}th batch is finished.')

        # pdb.set_trace()
        log_string = f"==> Epoch {epoch:04d} val results: \n"

        valid_loss=np.average(valid_losses)
        new_line = f"===> average loss: {valid_loss:.05f} \n"
        log_string += new_line
        print(log_string)

        if args.rank == 0 and not args.disable_wandb:
            wandb.log(
                {
                f"Fold {args.fold} - Validation Loss": valid_loss,
                "custom_step": niters,
                },
                # step=niters,
            )
        print("=> Finish Evaluating")
        return valid_loss

    @torch.no_grad()
    def test_metrics(self, epoch=0, niters=0):
        print("=> Start Testing")
        args = self.args
        model = self.wrapped_model
        val_loader = self.test_dataloader
        val_samples = len(val_loader)
        print(f"test samples: {val_samples}")

        # switch to evaluation mode
        model.eval()
        valid_losses = []
        predictions = []
        label_test = []

        for i, batch_data in enumerate(val_loader):
            if args.dataset in ['hospital','dzne', 'adni']:
                image = batch_data[0]
                target = batch_data[1]
                # print(image.shape)
            else:
                image = batch_data['image']
                target = batch_data['label']
            # image, target = batch_data['image'], batch_data['label']
            if args.gpu is not None:
                image = image.to(args.gpu, non_blocking=True)
                target = target.to(args.gpu, non_blocking=True)

            # get the validation loss
            outputs = F.softmax(model(image), dim=1)
            predictions.extend(torch.argmax(outputs, dim=1).tolist())
            label_test.extend(torch.argmax(target, dim=1).tolist())  # Accumulate true labels
            loss = self.loss_fn.ce(outputs, target)
            #TODO: need .item()?
            valid_losses.append(loss.item())

            # print(f'==> Testing on the {i+1}th batch is finished.')

        # pdb.set_trace()
        log_string = f"==> Fold {args.fold} test results: \n"

        valid_loss=np.average(valid_losses)
        # label_test = torch.argmax(target, dim=1).tolist()

        accuracy = accuracy_score(label_test, predictions)
        bal_acc = balanced_accuracy_score(label_test, predictions)
        precision = precision_score(label_test, predictions, average='macro')
        recall = recall_score(label_test, predictions, average='macro')
        f1 = f1_score(label_test, predictions, average='macro')

        # self.fivefolds_test_loss.append(valid_loss)
        # self.fivefolds_test_accuracy.append(accuracy)
        # self.fivefolds_test_bal_accuracy.append(bal_acc)
        # self.fivefold_test_precision.append(precision)
        # self.fivefold_test_recall.append(recall)
        # self.fivefold_test_f1.append(f1)

        log_string += f"===> Test loss: {valid_loss:.05f} \n "
        log_string += f"===> Accuracy: {accuracy:.05f} \n "
        log_string += f"===> Balanced Accuracy: {bal_acc:.05f} \n "
        log_string += f"===> Precision: {precision:.05f} \n "
        log_string += f"===> Recall: {recall:.05f} \n "
        log_string += f"===> F1-score: {f1:.05f} \n "

        print(log_string)

        return valid_loss, accuracy, bal_acc, precision, recall, f1


    def speedometerv2(self):
        args = self.args
        model = self.wrapped_model

        model.eval()

        time_meters = defaultdict(list)
        num_trials = 16
        for t in range(num_trials):
            image = torch.rand(args.batch_size, args.in_chans, args.roi_x, args.roi_y, args.roi_z)
            single_image = torch.rand(1, args.in_chans, args.roi_x, args.roi_y, args.roi_z)
            if args.gpu is not None:
                single_image = single_image.cuda(args.gpu, non_blocking=True) # [B, 4, IH, IW, ID]
                image = image.cuda(args.gpu, non_blocking=True) # [B, 4, IH, IW, ID]
            print(f"image shape is {image.shape}")
            if t == 0:
                try:
                    macs = profile_macs(model, single_image) * 1e-9
                except:
                    macs = -1
                print(f"MACS is {macs} G")
            # target = batch_data['label']
            if t > 2 and t < 13:
                start_time = time.perf_counter()
                model(image, time_meters=time_meters)
                torch.cuda.synchronize()
                end_time  = time.perf_counter()
                time_meters['total'].append(end_time - start_time)
            else:
                model(image)
            print(f"finish trial {t}")
        for key in time_meters.keys():
            # print(f'num of records in {key} is {len(time_meters[key])}')
            avg_time = np.mean(time_meters[key])
            print(f"=> averaged inference time for {key} is {avg_time}")
        print(f"MACS is {macs} G")
        print(f"{4 / np.mean(time_meters['total'])} total")
        print(f"{4 / np.mean(time_meters['enc'])} enc")
        # print(f"==> avg inference time over all trials is {np.mean(trials)}")

    def calc_sparsity(self):
        args = self.args
        raw_transform = get_raw_transforms(args)
        raw_dataloader = get_val_loader(args,
                                batch_size=1,
                                workers=1,
                                val_transform=raw_transform)
        if args.dataset == 'msd_brats':
            cls_sparsity_sum = [0] * 3
        elif args.dataset == 'btcv':
            cls_sparsity_sum = [0] * 14
        for b_idx, batch_data in enumerate(raw_dataloader):
            print("=========================================")
            print(f"the {b_idx} sample")
            # image = batch_data['image']
            target = batch_data['label']
            # pdb.set_trace()
            # print(f"{b_idx}th sample")
            volume_shape = target.shape
            num_voxels = volume_shape[-3] * volume_shape[-2] * volume_shape[-1]
            if args.dataset == 'msd_brats':
                for c in range(3):
                    cls_target = target[0, c]
                    num_target = cls_target.sum()
                    sparsity = (num_voxels - num_target) / num_voxels
                    cls_sparsity_sum[c] += sparsity
                    print("-----------------------")
                    print(f"current class {c} sparsity is ({num_voxels} - {num_target})/{num_voxels} = {sparsity}")
                    print(f"current class {c} accum sparsity is {cls_sparsity_sum[c]/(b_idx+1)}")
            elif args.dataset == 'btcv':
                for c in range(14):
                    if c == 0:
                        num_target = num_voxels - (target[0, 0] == c).sum()
                    else:
                        num_target = (target[0, 0] == c).sum()
                    sparsity = (num_voxels - num_target) / num_voxels
                    cls_sparsity_sum[c] += sparsity
                    print("-----------------------")
                    print(f"current class {c} sparsity is ({num_voxels} - {num_target})/{num_voxels} = {sparsity}")
                    print(f"current class {c} accum sparsity is {cls_sparsity_sum[c]/(b_idx+1)}")
                

    def resume(self):
        args = self.args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler']) # additional line compared with base imple
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    def adjust_learning_rate(self, epoch, args):
        """Base schedule: CosineDecay with warm-up."""
        init_lr = self.lr
        if epoch < args.warmup_epochs:
            cur_lr = init_lr * epoch / args.warmup_epochs
        else:
            cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if 'lr_scale' in param_group:
                param_group['lr'] = cur_lr * param_group['lr_scale']
            else:
                param_group['lr'] = cur_lr
