import os
import warnings

import torch
import torch.multiprocessing as mp
# torch.multiprocessing.set_sharing_strategy('file_system')
import wandb

import sys
sys.path.append('lib/')

from lib.utils import set_seed, dist_setup, get_conf
import lib.trainers as trainers

import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Total CUDA devices: ", torch.cuda.device_count())


def main():

    args = get_conf()

    args.test = False

    # set seed if required
    set_seed(args.seed)

    if not args.multiprocessing_distributed and args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, 
                nprocs=ngpus_per_node, 
                args=(args,))
    else:
        print("single process")
        main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    ngpus_per_node = args.ngpus_per_node
    dist_setup(ngpus_per_node, args)

    # init trainer
    trainer_class = getattr(trainers, f'{args.trainer_name}', None)
    assert trainer_class is not None, f"Trainer class {args.trainer_name} is not defined"
    trainer = trainer_class(args)

    if args.rank == 0 and not args.disable_wandb:
        if args.wandb_id is None:
            args.wandb_id = wandb.util.generate_id()

        run = wandb.init(project=f"{args.proj_name}_{args.dataset}", 
                        name=args.run_name, 
                        config=vars(args),
                        id=args.wandb_id,
                        resume='allow',
                        dir=args.output_dir)
        wandb.define_metric("custom_step")

    if args.fivefolds:
        fivefolds_test_loss = []
        fivefolds_test_accuracy = []
        fivefolds_test_bal_accuracy = []
        fivefold_test_precision = []
        fivefold_test_recall = []
        fivefold_test_f1 = []

        for f in range(1,6):
            args.fold = f

            if args.rank == 0 and not args.disable_wandb:
                # define which metrics will be plotted against it
                wandb.define_metric(f"Fold {f} - lr", step_metric="custom_step")
                wandb.define_metric(f"Fold {f} - Training Loss", step_metric="custom_step")
                wandb.define_metric(f"Fold {f} - Validation Loss", step_metric="custom_step")

            trainer = trainer_class(args)
            
            # create model
            trainer.build_model()
            # create optimizer
            trainer.build_optimizer()
            # resume training
            if args.resume:
                trainer.resume()
            trainer.build_dataloader()

            test_loss, accuracy, bal_acc, precision, recall, f1 = trainer.run()
        
            fivefolds_test_loss.append(test_loss)
            fivefolds_test_accuracy.append(accuracy)
            fivefolds_test_bal_accuracy.append(bal_acc)
            fivefold_test_precision.append(precision)
            fivefold_test_recall.append(recall)
            fivefold_test_f1.append(f1)

        print(f"Average Test Loss: {np.mean(fivefolds_test_loss)}, std: {np.std(fivefolds_test_loss)}")
        print(f"Average Test Accuracy: {np.mean(fivefolds_test_accuracy)}, std: {np.std(fivefolds_test_accuracy)}")
        print(f"Average Test Balanced Accuracy: {np.mean(fivefolds_test_bal_accuracy)}, std: {np.std(fivefolds_test_bal_accuracy)}")
        print(f"Average Test Precision: {np.mean(fivefold_test_precision)}, std: {np.std(fivefold_test_precision)}")
        print(f"Average Test Recall: {np.mean(fivefold_test_recall)}, std: {np.std(fivefold_test_recall)}")
        print(f"Average Test F1: {np.mean(fivefold_test_f1)}, std: {np.std(fivefold_test_f1)}")

    else: 
        # create model
        trainer.build_model()
        # create optimizer
        trainer.build_optimizer()
        # resume training
        if args.resume:
            trainer.resume()
        trainer.build_dataloader()

        trainer.run()

    if args.rank == 0 and not args.disable_wandb:
        run.finish()


if __name__ == '__main__':
    main()
