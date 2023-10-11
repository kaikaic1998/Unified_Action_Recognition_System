# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import os
import os.path as osp
import time
import torch
import torch.distributed as dist
from mmcv.engine import multi_gpu_test
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, EpochBasedRunner, OptimizerHook, build_optimizer, get_dist_info

from pyskl.core import DistEvalHook
from pyskl.datasets import build_dataloader, build_dataset
from pyskl.utils import cache_checkpoint, get_root_logger

# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: E722
import argparse
import mmcv
import os
import os.path as osp
import time
import torch
import torch.distributed as dist
from mmcv import Config
from mmcv import digit_version as dv
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

from pyskl import __version__
from pyskl.apis import init_random_seed, train_model
from pyskl.datasets import build_dataset
from pyskl.models import build_model
from pyskl.utils import collect_env, get_root_logger, mc_off, mc_on, test_port


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)

    dist.broadcast(random_num, src=0)
    return random_num.item()


def train_model(model,
                dataset,
                cfg,
                validate=False,
                test=dict(test_best=False, test_last=False),
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        validate (bool): Whether to do evaluation. Default: False.
        test (dict): The testing option, with two keys: test_last & test_best.
            The value is True or False, indicating whether to test the
            corresponding checkpoint.
            Default: dict(test_best=False, test_last=False).
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(log_level=cfg.get('log_level', 'INFO'))

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        persistent_workers=cfg.data.get('persistent_workers', False),
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))

    data_loaders = [
        build_dataloader(ds, **dataloader_setting) for ds in dataset
    ]

    # put model on gpus
    find_unused_parameters = cfg.get('find_unused_parameters', True)
    # Sets the `find_unused_parameters` parameter in
    # torch.nn.parallel.DistributedDataParallel
    # model = MMDistributedDataParallel(
    #     model.cuda(),
    #     device_ids=[torch.cuda.current_device()],
    #     broadcast_buffers=False,
    #     find_unused_parameters=find_unused_parameters)
    # model = model.to(torch.device('cuda:0'))

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    Runner = EpochBasedRunner
    runner = Runner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    if 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    runner.register_hook(DistSamplerSeedHook())

    eval_hook = None
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            persistent_workers=cfg.data.get('persistent_workers', False),
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEvalHook(val_dataloader, **eval_cfg)
        runner.register_hook(eval_hook)

    if cfg.get('resume_from', None):
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from', None):
        cfg.load_from = cache_checkpoint(cfg.load_from)
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)

    # dist.barrier()
    time.sleep(2)

    if test['test_last'] or test['test_best']:
        best_ckpt_path = None
        if test['test_best']:
            assert eval_hook is not None
            best_ckpt_path = None
            ckpt_paths = [x for x in os.listdir(cfg.work_dir) if 'best' in x]
            ckpt_paths = [x for x in ckpt_paths if x.endswith('.pth')]
            if len(ckpt_paths) == 0:
                logger.info('Warning: test_best set, but no ckpt found')
                test['test_best'] = False
                if not test['test_last']:
                    return
            elif len(ckpt_paths) > 1:
                epoch_ids = [
                    int(x.split('epoch_')[-1][:-4]) for x in ckpt_paths
                ]
                best_ckpt_path = ckpt_paths[np.argmax(epoch_ids)]
            else:
                best_ckpt_path = ckpt_paths[0]
            if best_ckpt_path:
                best_ckpt_path = osp.join(cfg.work_dir, best_ckpt_path)

        test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
        tmpdir = cfg.get('evaluation', {}).get('tmpdir', osp.join(cfg.work_dir, 'tmp'))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            persistent_workers=cfg.data.get('persistent_workers', False),
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('test_dataloader', {}))

        test_dataloader = build_dataloader(test_dataset, **dataloader_setting)

        names, ckpts = [], []

        if test['test_last']:
            names.append('last')
            ckpts.append(None)
        if test['test_best']:
            names.append('best')
            ckpts.append(best_ckpt_path)

        for name, ckpt in zip(names, ckpts):
            if ckpt is not None:
                runner.load_checkpoint(ckpt)

            outputs = multi_gpu_test(runner.model, test_dataloader, tmpdir)
            rank, _ = get_dist_info()
            if rank == 0:
                out = osp.join(cfg.work_dir, f'{name}_pred.pkl')
                test_dataset.dump_results(outputs, out)

                eval_cfg = cfg.get('evaluation', {})
                for key in [
                        'interval', 'tmpdir', 'start',
                        'save_best', 'rule', 'by_epoch', 'broadcast_bn_buffers'
                ]:
                    eval_cfg.pop(key, None)

                eval_res = test_dataset.evaluate(outputs, **eval_cfg)
                logger.info(f'Testing results of the {name} checkpoint')
                for metric_name, val in eval_res.items():
                    logger.info(f'{metric_name}: {val:.04f}')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    # parser.add_argument('config', help='train config file path')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--test-last',
        action='store_true',
        help='whether to test the checkpoint after training')
    parser.add_argument(
        '--test-best',
        action='store_true',
        help='whether to test the best checkpoint (if applicable) after training')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['pytorch', 'slurm'],
        default='pytorch',
        help='job launcher')
    parser.add_argument(
        '--compile',
        action='store_true',
        help='whether to compile the model before training / testing (only available in pytorch 2.0)')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--local-rank', type=int, default=-1)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    print('args: ', args)

    # config = configs/stgcn++/stgcn++_ntu120_xset_hrnet/j.py
    cfg = Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority:
    # config file > default (base filename)
    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    if not hasattr(cfg, 'dist_params'):
        cfg.dist_params = dict(backend='nccl')

    # init_dist(args.launcher, **cfg.dist_params)
    rank, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)

    auto_resume = cfg.get('auto_resume', True)
    if auto_resume and cfg.get('resume_from', None) is None:
        resume_pth = osp.join(cfg.work_dir, 'latest.pth')
        if osp.exists(resume_pth):
            cfg.resume_from = resume_pth

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.get('log_level', 'INFO'))

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Config: {cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)

    cfg.seed = seed
    meta['seed'] = seed
    meta['config_name'] = osp.basename(args.config)
    meta['work_dir'] = osp.basename(cfg.work_dir.rstrip('/\\'))

    print('\nmodel: ', cfg.model)
    model = build_model(cfg.model) # after this line, model is the actual model structure
    # can just fine-tune the last classification layer
    print('\nmodel after: ', model, '\n')

    if dv(torch.__version__) >= dv('2.0.0') and args.compile:
        model = torch.compile(model)

    print('cfg.data.train: ', cfg.data.train)
    datasets = [build_dataset(cfg.data.train)]

    cfg.workflow = cfg.get('workflow', [('train', 1)])
    assert len(cfg.workflow) == 1
    if cfg.checkpoint_config is not None:
        # save pyskl version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            pyskl_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text)

    #                              False                       False
    test_option = dict(test_last=args.test_last, test_best=args.test_best)

    default_mc_cfg = ('localhost', 22077)
    memcached = cfg.get('memcached', False)

    if rank == 0 and memcached:
        # mc_list is a list of pickle files you want to cache in memory.
        # Basically, each pickle file is a dictionary.
        mc_cfg = cfg.get('mc_cfg', default_mc_cfg)
        assert isinstance(mc_cfg, tuple) and mc_cfg[0] == 'localhost'
        if not test_port(mc_cfg[0], mc_cfg[1]):
            mc_on(port=mc_cfg[1], launcher=args.launcher)
        retry = 3
        while not test_port(mc_cfg[0], mc_cfg[1]) and retry > 0:
            time.sleep(5)
            retry -= 1
        assert retry >= 0, 'Failed to launch memcached. '

    # dist.barrier()

    train_model(model, datasets, cfg, validate=args.validate, test=test_option, timestamp=timestamp, meta=meta)
    # dist.barrier()

    if rank == 0 and memcached:
        mc_off()


if __name__ == '__main__':
    main()
