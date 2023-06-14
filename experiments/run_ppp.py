import torch.multiprocessing as mp
# if __name__ == "__main__":
#     mp.set_start_method('spawn')
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

import argparse
from copy import deepcopy
from datetime import datetime
from glob import glob
import fnmatch
import functools
import importlib
import itertools
import logging

import operator
import os
import queue
import random
import runpy
import shutil
import sys
import time

import h5py
import zarr
from joblib import Parallel, delayed
from natsort import natsorted
import numpy as np
import toml
import json
from git import Repo

from PatchPerPix import util
from PatchPerPix.evaluate import evaluate_patch, evaluate_numinst, evaluate_fg

from PatchPerPix import vote_instances as vi
from evaluateInstanceSegmentation import evaluate_file, summarize_metric_dict
from PatchPerPix.visualize import visualize_patches, visualize_instances


def merge_dicts(sink, source):
    if not isinstance(sink, dict) or not isinstance(source, dict):
        raise TypeError('Args to merge_dicts should be dicts')

    for k, v in source.items():
        if isinstance(source[k], dict) and isinstance(sink.get(k), dict):
            sink[k] = merge_dicts(sink[k], v)
        else:
            sink[k] = v

    return sink


def backup_and_copy_file(source, target, fn):
    target = os.path.join(target, fn)
    if os.path.exists(target):
        os.replace(target, target + "_backup" + str(int(time.time())))
    if source is not None:
        source = os.path.join(source, fn)
        shutil.copy2(source, target)

def check_file(fn, remove_on_error=False, key=None):
    if fn.endswith("zarr"):
        try:
            fl = zarr.open(fn, 'r')
            if key is not None:
                tmp = fl[key]
            return True
        except Exception as e:
            logger.info("%s", e)
            if remove_on_error:
                shutil.rmtree(fn, ignore_errors=True)
            return False
    elif fn.endswith("hdf"):
        try:
            with h5py.File(fn, 'r') as fl:
                if key is not None:
                    tmp = fl[key]
            return True
        except Exception as e:
            if remove_on_error:
                os.remove(fn)
            return False
    else:
        raise NotImplementedError("invalid file type")

def time_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = datetime.now()
        ret = func(*args, **kwargs)
        logger.info('time %s: %s', func.__name__, str(datetime.now() - t0))
        return ret

    return wrapper


def fork(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logger.info("forking %s", func)
            p = mp.Process(target=func, args=args, kwargs=kwargs)
            p.start()
            p.join()
            if p.exitcode != 0:
                raise RuntimeError("child process died")
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            p.terminate()
            p.join()
            os._exit(-1)

    return wrapper


def fork_return(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logger.info("forking %s", func)
            q = mp.Queue()
            p = mp.Process(target=func,
                           args=args + (q,), kwargs=kwargs)
            p.start()
            results = None
            while p.is_alive():
                try:
                    results = q.get_nowait()
                except queue.Empty:
                    time.sleep(2)
            if p.exitcode == 0 and results is None:
                results = q.get()
            p.join()
            if p.exitcode != 0:
                raise RuntimeError("child process died")
            return results
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            p.terminate()
            p.join()
            os._exit(-1)

    return wrapper


logger = logging.getLogger(__name__)


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', action='append',
                        help=('Configuration files to use. For defaults, '
                              'see `config/default.toml`.'))
    parser.add_argument('-a', '--app', dest='app', required=True,
                        help=('Application to use. Choose out of cityscapes, '
                              'flylight, kaggle, etc.'))
    parser.add_argument('-r', '--root', dest='root', default=None,
                        help='Experiment folder to store results.')
    parser.add_argument('-s', '--setup', dest='setup', default=None,
                        help='Setup for experiment.', required=True)
    parser.add_argument('-id', '--exp-id', dest='expid', default=None,
                        help='ID for experiment.')
    parser.add_argument('--comment', default=None,
                        help='Note that will be printed in log')

    # action options
    parser.add_argument('-d', '--do', dest='do', default=[], nargs='+',
                        choices=['all',
                                 'mknet',
                                 'train',
                                 'predict',
                                 'decode',
                                 'label',
                                 'infer',
                                 'validate_checkpoints',
                                 'validate',
                                 'postprocess',
                                 'evaluate',
                                 'cross_validate',
                                 'visualize',
                                 'cleanup'
                                 ],
                        help='Task to do for experiment.')

    parser.add_argument('--test-checkpoint', dest='test_checkpoint',
                        default='last', choices=['last', 'best'],
                        help=('Specify which checkpoint to use for testing. '
                              'Either last or best (checkpoint validation).'))

    parser.add_argument('--checkpoint', dest='checkpoint', default=None,
                        type=int,
                        help='Specify which checkpoint to use.')

    parser.add_argument("--run_from_exp", action="store_true",
                        help='run from setup or from experiment folder')
    parser.add_argument("--validate_on_train", action="store_true",
                        help=('validate using training data'
                              '(to check for overfitting)'))
    parser.add_argument("--test_on_train", action="store_true",
                        help=('test using training data'
                              '(to check for overfitting)'))

    # train / val / test datasets
    parser.add_argument('--input-format', dest='input_format',
                        choices=['hdf', 'zarr', 'n5', 'tif'],
                        help='File format of dataset.')
    parser.add_argument('--train-data', dest='train_data', default=None,
                        help='Train dataset to use.')
    parser.add_argument('--val-data', dest='val_data', default=None,
                        help='Validation dataset to use.')
    parser.add_argument('--test-data', dest='test_data', default=None,
                        help='Test dataset to use.')

    # parameters for vote instances
    parser.add_argument('--vote-instances-cuda', dest='vote_instances_cuda',
                        action='store_true',
                        help='Determines if CUDA should be used to process '
                             'vote instances.')
    parser.add_argument('--vote-instances-blockwise',
                        dest='vote_instances_blockwise',
                        action='store_true',
                        help='Determines if vote instances should be '
                             'processed blockwise.')

    parser.add_argument("--debug_args", action="store_true",
                        help=('Set some low values to certain'
                              ' args for debugging.'))

    parser.add_argument("--predict_single", action="store_true",
                        help=('predict a single sapmle, for testing'))
    parser.add_argument("--term_after_patch_graph", action="store_true",
                        help=('terminate after patch graph, to split into GPU and CPU parts'))
    parser.add_argument("--graph_to_inst", action="store_true",
                        help=('only do patch graph to inst part of vote_instances'))
    parser.add_argument('--sample', default=None,
                        help='Sample to process.')
    parser.add_argument("--skip_predict", action="store_true",
                        help=('skip prediction'
                              'e.g. during validate_checkpoints.'))
    parser.add_argument("--only_predict_decode", action="store_true",
                        help=('only prediction and decode'
                              'e.g. during validate_checkpoints.'))
    parser.add_argument("--skip_evaluate", action="store_true",
                        help=('skip evaluation'))
    parser.add_argument("--add_partly_val", action="store_true",
                        help=('add partly labeled data to validation set'))
    parser.add_argument('--val_id', default=-1, type=int,
                        help='id of val params to process')
    parser.add_argument("--batched", action="store_true",
                        help='train with batch size > 1')
    parser.add_argument("--gp_predict", action="store_true",
                        help='predict with gunpowder')
    parser.add_argument("--predict_monai", action="store_true",
                        help='predict with monai (overlap+blend)')

    args = parser.parse_args()

    return args


def create_folders(args, filebase):
    # create experiment folder
    os.makedirs(filebase, exist_ok=True)

    if args.expid is None and args.run_from_exp:
        setup = os.path.join(args.app, 'setups', args.setup)
        backup_and_copy_file(setup, filebase, 'train.py')
        backup_and_copy_file(setup, filebase, 'mknet.py')
        try:
            backup_and_copy_file(setup, filebase, 'torch_loss.py')
        except FileNotFoundError:
            pass
        try:
            backup_and_copy_file(setup, filebase, 'torch_model.py')
        except FileNotFoundError:
            pass
        try:
            backup_and_copy_file(setup, filebase, 'predict.py')
        except FileNotFoundError:
            pass
        try:
            backup_and_copy_file(setup, filebase, 'predict_no_gp.py')
        except FileNotFoundError:
            pass
        try:
            backup_and_copy_file(setup, filebase, 'predict_monai.py')
        except FileNotFoundError:
            pass
        try:
            backup_and_copy_file(setup, filebase, 'label.py')
        except FileNotFoundError:
            pass
        try:
            backup_and_copy_file(setup, filebase, 'decode.py')
        except FileNotFoundError:
            pass

    # create train folders
    train_folder = os.path.join(filebase, 'train')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'snapshots'), exist_ok=True)

    # create val folders
    if args.validate_on_train:
        val_folder = os.path.join(filebase, 'val_train')
    else:
        val_folder = os.path.join(filebase, 'val')
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(os.path.join(val_folder, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(val_folder, 'instanced'), exist_ok=True)

    # create test folders
    if args.test_on_train:
        test_folder = os.path.join(filebase, 'test_train')
    else:
        test_folder = os.path.join(filebase, 'test')
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(os.path.join(test_folder, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(test_folder, 'instanced'), exist_ok=True)

    return train_folder, val_folder, test_folder


def update_config(args, config):
    if args.train_data is not None:
        config['data']['train_data'] = args.train_data

    if args.val_data is not None:
        config['data']['val_data'] = args.val_data

    if args.test_data is not None:
        config['data']['test_data'] = args.test_data

    if args.input_format is not None:
        config['data']['input_format'] = args.input_format
    if 'input_format' not in config['data']:
        raise ValueError("Please provide data/input_format in cl or config")

    if args.validate_on_train:
        config['data']['validate_on_train'] = True
        config['data']['val_data'] = config['data']['train_data']
    else:
        config['data']['validate_on_train'] = False

    if args.test_on_train:
        config['data']['test_on_train'] = True
        config['data']['test_data'] = config['data']['train_data']
    else:
        config['data']['test_on_train'] = False

    if args.vote_instances_cuda:
        config['vote_instances']['cuda'] = True

    if args.vote_instances_blockwise:
        config['vote_instances']['blockwise'] = True


def setDebugValuesForConfig(config):
    config['training']['max_iterations'] = 10
    config['training']['checkpoints'] = 10
    config['training']['snapshots'] = 10
    config['training']['profiling'] = 10
    config['training']['num_workers'] = 1
    config['training']['cache_size'] = 1


@fork
@time_func
def mknet(args, config, train_folder, test_folder):
    if args.run_from_exp:
        mk_net_fn = runpy.run_path(
            os.path.join(config['base'], 'mknet.py'))['mk_net']
        mk_net_fn_train = mk_net_fn
    else:
        mk_net_fn = importlib.import_module(
            args.app + '.setups.' + args.setup + '.mknet').mk_net
        if args.batched:
            mk_net_fn_train = importlib.import_module(
                args.app + '.setups.' + args.setup + '.mknet_bs').mk_net
        else:
            mk_net_fn_train = mk_net_fn

    mk_net_fn_train(name=config['model']['train_net_name'],
                    input_shape=config['model']['train_input_shape'],
                    output_folder=train_folder,
                    autoencoder=config.get('autoencoder'),
                    **config['data'],
                    **config['model'],
                    **config['optimizer'],
                    batch_size=config['training']['batch_size'],
                    debug=config['general']['debug'])
    mk_net_fn(name=config['model']['test_net_name'],
              input_shape=config['model']['test_input_shape'],
              output_folder=test_folder,
              autoencoder=config.get('autoencoder'),
              **config['data'],
              **config['model'],
              **config['optimizer'],
              batch_size=config['training']['batch_size'],
              debug=config['general']['debug'])


# @fork
@time_func
def train(args, config, train_folder):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise RuntimeError("no free GPU available!")
    # if __name__ == "__main__" and config["training"].get("add_affinities", "cpu") == "torch":
        # mp.set_start_method('spawn')

    data_files = get_list_train_files(config)
    val_files = get_list_train_files(config, val=True)
    if args.run_from_exp:
        train_fn = runpy.run_path(
            os.path.join(config['base'], 'train.py'))['train_until']
    elif args.batched:
        train_fn = importlib.import_module(
            args.app + '.setups.' + args.setup + '.train_bs').train_until
    else:
        train_fn = importlib.import_module(
            args.app + '.setups.' + args.setup + '.train').train_until

    train_fn(name=config['model']['train_net_name'],
             max_iteration=config['training']['max_iterations'],
             output_folder=train_folder,
             data_files=data_files,
             val_files=val_files,
             **config['data'],
             **config['model'],
             **config['training'],
             **config['optimizer'],
             **config.get('preprocessing', {}))


def get_list_train_files(config, val=False):
    if val:
        data = config['data']['val_data']
    else:
        data = config['data']['train_data']
    if os.path.isfile(data):
        files = [data]
    elif os.path.isdir(data):
        if 'folds' in config['training']:
            files = glob(os.path.join(
                data + "_folds" + config['training']['folds'],
                "*." + config['data']['input_format']))
        elif config['data'].get('sub_folders', False):
            files = glob(os.path.join(
                data, "*", "*." + config['data']['input_format']))
        else:
            files = glob(os.path.join(data,
                                      "*." + config['data']['input_format']))
    else:
        raise ValueError(
            "please provide file or directory for data/train_data", data)
    return files


def get_list_samples(config, data, file_format, filter=None):
    logger.info("reading data from %s", data)
    # read data
    if os.path.isfile(data):
        if file_format == ".hdf" or file_format == "hdf":
            with h5py.File(data, 'r') as f:
                samples = [k for k in f]
        else:
            NotImplementedError("Add reader for %s format",
                                os.path.splitext(data)[1])
    elif data.endswith('.zarr'):
        samples = [os.path.basename(data).split(".")[0]]
    elif os.path.isdir(data):
        samples = fnmatch.filter(os.listdir(data),
                                 '*.' + file_format)
        samples = [os.path.splitext(s)[0] for s in samples]
        if not samples:
            for d in os.listdir(data):
                tmp = fnmatch.filter(os.listdir(os.path.join(data, d)),
                                     '*.' + file_format)
                tmp = [os.path.join(d, os.path.splitext(s)[0]) for s in tmp]
                samples += tmp
    else:
        logger.info("%s %s %s", data, file_format, filter)
        raise NotImplementedError("Data must be file or directory")
    logger.debug(samples)

    # read filter list
    if filter is not None:
        if os.path.isfile(filter):
            if filter.endswith(".hdf"):
                with h5py.File(filter, 'r') as f:
                    filter_list = [k for k in f]
            else:
                NotImplementedError("Add reader for %s format",
                                    os.path.splitext(data)[1])
        elif filter.endswith('.zarr'):
            filter_list = [os.path.basename(filter).split(".")[0]]
        elif os.path.isdir(filter):
            filter_list = fnmatch.filter(os.listdir(filter), '*')
            filter_list = [os.path.splitext(s)[0] for s in filter_list]
            if not filter_list:
                for d in os.listdir(filter):
                    tmp = fnmatch.filter(os.listdir(os.path.join(filter, d)),
                                         '*.' + file_format)
                    tmp = [os.path.join(d, os.path.splitext(s)[0]) for s in tmp]
                    filter_list += tmp
        else:
            logger.info(
                "%s %s %s %s %s %s %s", data, file_format, filter,
                os.path.isfile(filter), os.path.isdir(filter),
                os.path.isfile(data), os.path.isdir(data))
            raise NotImplementedError("Data must be file or directory")
        print(filter_list[:5])
        samples = [s for s in samples if s in filter_list]
    print(samples[:5])
    return samples


@fork
@time_func
def predict_sample(args, config, name, data, sample, checkpoint, input_folder,
                   output_folder):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise RuntimeError("no free GPU available!")

    if args.run_from_exp:
        predict_fn = runpy.run_path(
            os.path.join(config['base'], 'predict.py'))['predict']
    else:
        predict_fn = importlib.import_module(
            args.app + '.setups.' + args.setup + '.predict').predict

    logger.info('predicting %s!', sample)
    predict_fn(name=name, sample=sample, checkpoint=checkpoint,
               data_folder=data, input_folder=input_folder,
               output_folder=output_folder,
               **config['data'],
               **config['model'],
               **config.get('preprocessing', {}),
               **config['prediction'])


@fork
@time_func
def predict_autoencoder(args, config, data, checkpoint, train_folder,
                        output_folder):
    import tensorflow as tf

    if args.run_from_exp:
        eval_predict_fn = runpy.run_path(
            os.path.join(config['base'], 'eval_predict.py'))['run']
    else:
        eval_predict_fn = importlib.import_module(
            args.app + '.setups.' + args.setup + '.eval_predict').run

    input_shape = tuple(p for p in config['model']['patchshape'] if p > 1)
    checkpoint_file = get_checkpoint_file(checkpoint,
                                          config['model']['train_net_name'],
                                          train_folder)
    samples = get_list_samples(config, data, config['data']['input_format'])
    if not os.path.isfile(data):
        for idx, s in enumerate(samples):
            samples[idx] = os.path.join(data,
                                        s + "." + config['data']['input_format'])

    eval_predict_fn(mode=tf.estimator.ModeKeys.PREDICT,
                    input_shape=input_shape,
                    max_samples=32,
                    checkpoint_file=checkpoint_file,
                    output_folder=output_folder,
                    samples=samples,
                    **config['model'],
                    **config['data'],
                    **config['prediction'],
                    **config['visualize'])


@fork
@time_func
def predict_no_gp(args, config, name, data, samples, checkpoint, input_folder,
                  output_folder):
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise RuntimeError("no free GPU available!")

    logger.info('predicting!')

    if not args.predict_monai:
        if args.run_from_exp:
            predict_fn = runpy.run_path(
                os.path.join(config['base'], 'predict_no_gp.py'))['predict']
        else:
            predict_fn = importlib.import_module(
                args.app + '.setups.' + args.setup + '.predict_no_gp').predict
    else:
        if args.run_from_exp:
            predict_fn = runpy.run_path(
                os.path.join(config['base'], 'predict_monai.py'))['predict']
        else:
            predict_fn = importlib.import_module(
                args.app + '.setups.' + args.setup + '.predict_monai').predict
    predict_fn(name=name, samples=samples, checkpoint=checkpoint,
               data_folder=data, input_folder=input_folder,
               output_folder=output_folder,
               **config['data'],
               **config['model'],
               batch_size=config['training']['batch_size'],
               **config.get('preprocessing', {}),
               **config['prediction'])


@time_func
def predict(args, config, name, data, checkpoint, test_folder, output_folder):
    if data.endswith("npy"):
        samples = [data]
    else:
        samples = get_list_samples(config, data, config['data']['input_format'])
        if config["data"].get("add_partly_val", False):
            samples += get_list_samples(
                    config, data.replace("complete", "partly"),
                    config["data"]["input_format"]
                    )

    if args.sample is not None:
        samples = [s for s in samples if args.sample in s]

    samplesT = []
    for idx, sample in enumerate(samples):
        fl = os.path.join(output_folder,
                          sample + '.' + config['prediction']['output_format'])
        if not config['general']['overwrite'] and os.path.exists(fl):
            key = ('aff_key'
                   if not (config['training'].get('train_code') or
                           config['model'].get('train_code'))
                   else 'code_key')
            if check_file(
                    fl, remove_on_error=False,
                    key=config['prediction'].get(key, "volumes/pred_affs")):
                logger.info('Skipping prediction for %s. Already exists!',
                            sample)
                if args.predict_single:
                    break
                else:
                    continue
            else:
                logger.info('prediction %s broken. recomputing..!',
                            sample)
        samplesT.append(sample)
    samples = samplesT

    if (not args.gp_predict or args.predict_monai) and samples:
        predict_no_gp(args, config, name, data, samples, checkpoint,
                      test_folder, output_folder)
        return

    for idx, sample in enumerate(samples):
        print("predicting with output %s" % os.path.join(
                output_folder,
                sample + '.' + config['prediction']['output_format']))
        if args.debug_args and idx >= 2:
            break

        predict_sample(args, config, name, data, sample, checkpoint,
                       test_folder, output_folder)
        if args.predict_single:
            break


@fork
@time_func
def decode(args, config, data, checkpoint, pred_folder, output_folder):
    in_format = config['prediction']['output_format']
    # samples = get_list_samples(config, pred_folder, in_format, data)
    samples = get_list_samples(config, pred_folder, in_format)

    if args.sample is not None:
        samples = [s for s in samples if args.sample in s]

    to_be_skipped = []
    for sample in samples:
        pred_file = os.path.join(output_folder, sample + '.' + in_format)
        if not config['general']['overwrite'] and os.path.exists(pred_file):
            if check_file(pred_file, remove_on_error=False,
                          key=config['prediction'].get('aff_key',
                                                       "volumes/pred_affs")):
                logger.info('Skipping decoding for %s. Already exists!', sample)
                to_be_skipped.append(sample)
    for sample in to_be_skipped:
        samples.remove(sample)
    if len(samples) == 0:
        return

    logger.info("Decoding still to be done for: %s", samples)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise RuntimeError("no free GPU available!")
    try:
        import tensorflow as tf
        mode = tf.estimator.ModeKeys.PREDICT
    except:
        logger.info(
            "unable to \"import tensorflow\" in def decode, "
            "this is ok as long as pytorch is used anyway.")
        mode = None
    for idx, s in enumerate(samples):
        samples[idx] = os.path.join(pred_folder, s + "." + in_format)

    if args.run_from_exp:
        decode_fn = runpy.run_path(
            os.path.join(config['base'], 'decode.py'))['decode']
    else:
        decode_fn = importlib.import_module(
            args.app + '.setups.' + args.setup + '.decode').decode

    if config['model'].get('code_units'):
        input_shape = (config['model'].get('code_units'),)
    else:
        input_shape = None

    decode_fn(
        mode=mode,
        input_shape=input_shape,
        checkpoint_file=checkpoint,
        output_folder=output_folder,
        samples=samples,
        included_ae_config=config.get('autoencoder'),
        **config['model'],
        **config['prediction'],
        **config['visualize'],
        **config['data'],
        batch_size=config['training']['batch_size'],
        num_parallel_samples=config['vote_instances']['num_parallel_samples']
    )


def get_checkpoint_file(iteration, name, train_folder):
    return os.path.join(train_folder, name + '_checkpoint_%d' % iteration)


def get_checkpoint_list(name, train_folder):
    checkpoints = natsorted(glob(
        os.path.join(train_folder, name + '_checkpoint_*.index')))
    return [int(os.path.splitext(os.path.basename(cp))[0].split("_")[-1])
            for cp in checkpoints]


def select_validation_data(config, train_folder, val_folder):
    if config['data'].get('validate_on_train'):
        if 'folds' in config['training']:
            data = config['data']['train_data'] + \
                   "_folds" + str(config['training']['folds'])
        else:
            data = config['data']['train_data']
        output_folder = train_folder
    else:
        if 'fold' in config['validation']:
            data = config['data']['val_data'] + \
                   "_fold" + str(config['validation']['fold'])
        else:
            data = config['data']['val_data']
        output_folder = val_folder
    return data, output_folder


@time_func
def validate_checkpoint(args, config, data, checkpoint, params, train_folder,
                        test_folder, output_folder):
    logger.info("validating checkpoint %d %s", checkpoint, params)

    # create test iteration folders
    params_str = [k + "_" + str(v).replace(".", "_").replace(
        " ", "").replace(",", "_").replace("[", "_").replace(
            "]", "_").replace("(", "_").replace(")", "_")
                  for k, v in params.items()]
    pred_folder = os.path.join(output_folder, 'processed', str(checkpoint))
    inst_folder = os.path.join(output_folder, 'instanced', str(checkpoint),
                               *params_str)
    eval_folder = os.path.join(output_folder, 'evaluated', str(checkpoint),
                               *params_str)

    os.makedirs(pred_folder, exist_ok=True)
    os.makedirs(inst_folder, exist_ok=True)
    os.makedirs(eval_folder, exist_ok=True)

    # predict val data
    checkpoint_file = get_checkpoint_file(checkpoint,
                                          config['model']['train_net_name'],
                                          train_folder)
    logger.info("predicting checkpoint %d", checkpoint)
    # predict and evaluate autoencoder separately
    if args.app == "autoencoder":
        metric = evaluate_autoencoder(args, config,
                                      config['data']['test_data'],
                                      checkpoint, train_folder, eval_folder)
        logger.info("%s checkpoint %6d: %.4f (%s)",
                    config['evaluation']['metric'], checkpoint, metric, params)
        return metric
    # predict other apps
    if not args.skip_predict:
        predict(args, config, config['model']['test_net_name'], data,
                checkpoint_file, test_folder, pred_folder)

    # if config['evaluation'].get('prediction_only_test') and config['evaluation']['prediction'].get('eval_numinst_prediction'):
    #     eval_folder = os.path.join(output_folder, "evaluated", str(checkpoint))
    #     logger.info("evaluating prediction checkpoint %d", checkpoint)
    #     return evaluate_prediction(
    #         args, config, data, pred_folder, eval_folder)

    # if ppp learns code
    if config['training'].get('train_code') or \
       config['model'].get('train_code'):
        autoencoder_chkpt = config['model']['autoencoder_chkpt']
        if autoencoder_chkpt == "this":
            autoencoder_chkpt = checkpoint_file
        decode(args, config, data, autoencoder_chkpt, pred_folder, pred_folder)

    if args.only_predict_decode:
        return None

    if config['evaluation'].get('prediction_only'):
        eval_folder = os.path.join(output_folder, "evaluated", str(checkpoint))
        logger.info("evaluating prediction checkpoint %d", checkpoint)
        return evaluate_prediction(
            args, config, data, pred_folder, eval_folder)

    # vote instances
    logger.info("vote_instances checkpoint %d %s", checkpoint, params)
    vote_instances(args, config, data, pred_folder, inst_folder)
    if args.term_after_patch_graph:
        exit(0)

    # evaluate
    if not args.skip_evaluate:
        logger.info("evaluating checkpoint %d %s", checkpoint, params)
        metric = evaluate(args, config, data, inst_folder, eval_folder)
        logger.info("%s checkpoint %6d: " + ("%s" if isinstance(metric, dict) else "%.4f") + " (%s)",
                    config['evaluation']['metric'], checkpoint, metric, params)
    else:
        metric = None
    return metric


def get_postprocessing_params(config, params_product_list,
                              params_zip_list, test_config):
    params_product = {}
    for p in params_product_list:
        if config is None or config[p] == []:
            params_product[p] = [test_config[p]]
        else:
            params_product[p] = config[p]

    params_zip = {}
    for p in params_zip_list:
        if config is None or config[p] == []:
            params_zip[p] = [test_config[p]]
        else:
            params_zip[p] = config[p]

    return params_zip, params_product


def named_product(**items):
    if items:
        names = items.keys()
        vals = items.values()
        for res in itertools.product(*vals):
            yield dict(zip(names, res))
    else:
        yield {}

def named_zip(**items):
    if items:
        names = items.keys()
        vals = items.values()
        for res in zip(*vals):
            yield dict(zip(names, res))
    else:
        yield {}


def named_params(params_zip, params_product):
    logger.info("zip params %s", params_zip)
    logger.info("product params %s", params_product)
    if not params_product and params_zip:
        yield from named_zip(**params_zip)
    elif params_product and not params_zip:
        yield from named_product(**params_product)
    elif params_product and params_zip:
        names_product = params_product.keys()
        vals_product = params_product.values()
        names_zip = params_zip.keys()
        vals_zip = params_zip.values()

        for res in zip(*vals_zip):
            param_zip = dict(zip(names_zip, res))
            logger.info("%s", param_zip)
            for res in itertools.product(*vals_product):
                param_product = dict(zip(names_product, res))
                logger.info("%s", param_product)
                param_zip_tmp = deepcopy(param_zip)
                yield merge_dicts(param_zip_tmp, param_product)
    else:
        yield {}


def validate_checkpoints(args, config, data, checkpoints, train_folder,
                         test_folder, output_folder):
    # validate all checkpoints and return best one
    metrics = []
    ckpts = []
    params = []
    results = []
    if config['evaluation'].get('prediction_only'):
        for checkpoint in checkpoints:
            param_set = {}
            metric, ths = validate_checkpoint(args, config, data,
                                              checkpoint, param_set,
                                              train_folder, test_folder,
                                              output_folder)
            metrics.append(metric)
        for idx, checkpoint in enumerate(checkpoints):
            logger.info("%s checkpoint %6d:",
                        config['evaluation']['metric'], checkpoint)
            logger.info("%s (%s)", metrics[idx], ths)
            logger.info("best: %.4f at threshold %s",
                        np.max(metrics[idx]), ths[np.argmax(metrics[idx])])
        return None, None
    else:
        param_sets = list(named_params(
            *get_postprocessing_params(
                config['validation'],
                config['validation'].get(
                    'params_product',
                    config['validation'].get('params', [])),
                config['validation'].get('params_zip', []),
                config['vote_instances']
            )))
        logger.info("val params %s", param_sets)

    for checkpoint in checkpoints:
        logger.info("validating checkpoint %s", checkpoint)
        for idx, param_set in enumerate(param_sets):
            if args.val_id >= 0 and args.val_id != idx:
                continue
            val_config = deepcopy(config)
            for k in param_set.keys():
                val_config['vote_instances'][k] = param_set[k]

            if 'filterSzs' in config['validation']:
                filterSzs = config['validation']['filterSzs']
            elif 'filterSz' in config['evaluation']:
                filterSzs = [config['evaluation']['filterSz']]
            else:
                filterSzs = [None]

            if 'res_keys' in config['validation']:
                res_keys = config['validation']['res_keys']
            else:
                res_keys = [config['evaluation']['res_key']]

            eval_params = list(named_product(filterSz=filterSzs,
                                             res_key=res_keys))
            for eval_param in eval_params:
                logger.info("eval params %s", eval_param)
                val_config['evaluation']['res_key'] = eval_param['res_key']
                val_config['evaluation']['filterSz'] = eval_param['filterSz']
                metric = validate_checkpoint(
                    args, val_config, data, checkpoint,
                    param_set, train_folder, test_folder,
                    output_folder)
                metrics.append(metric)
                ckpts.append(checkpoint)
                tmp_param_set = deepcopy(param_set)
                tmp_param_set['filterSz'] = eval_param['filterSz']
                tmp_param_set['res_key'] = eval_param['res_key']
                params.append(tmp_param_set)
                results.append({'checkpoint': checkpoint,
                                'metric': str(metric),
                                'params': tmp_param_set})
                if metric is None:
                    continue
                logger.info("%s checkpoint %6d: " +
                            ("%s" if isinstance(metric, dict) else "%.4f") +
                            " (%s)",
                            config['evaluation']['metric'], checkpoint,
                            metric, tmp_param_set)

    if args.only_predict_decode:
        return None, None

    if val_config['evaluation'].get('prediction_only'):
        config['evaluation']['metric'] = '1_f1'

    for ch, metric, p in zip(ckpts, metrics, params):
        logger.info("%s checkpoint %6d: " +
                    ("%s" if isinstance(metric, dict) else "%.4f") + " (%s)",
                    config['evaluation']['metric'], ch, np.mean(metric), p)

    if config['general']['debug'] and None in metrics:
        logger.error("None in checkpoint found: %s (continuing with last)",
                     tuple(metrics))
        best_checkpoint = ckpts[-1]
        best_params = params[-1]
    else:
        if isinstance(metrics, list):
            met = []
            for ch, metric, p in zip(ckpts, metrics, params):
                try:
                    m = metric[config['evaluation']['metric']]
                except:
                    m = metric
                logger.info(
                    "%s checkpoint %6d: " +
                    ("%s" if isinstance(metric, dict) else "%.4f") + " (%s)",
                    config['evaluation']['metric'], ch,
                    m, p)
                met.append(np.mean(metric))
            metrics = met
            best_checkpoint = ckpts[np.argmax(metrics)]
            best_params = params[np.argmax(metrics)]
        elif isinstance(metrics, dict):
            mm = 0
            mmi = 0
            for i, (c, p, m) in enumerate(zip(ckpts, params, metrics)):
                if m["pq"] > mm:
                    mm = m["pq"]
                    mmi = i
                print(c, p, m)
            best_checkpoint = ckpts[i]
            best_params = params[i]
        else:
            best_checkpoint = ckpts[np.argmax(metrics)]
            best_params = params[np.argmax(metrics)]
    logger.info('best checkpoint: %d', best_checkpoint)
    logger.info('best params: %s', best_params)
    with open(os.path.join(output_folder, "results.json"), 'w') as f:
        json.dump(results, f)
    return best_checkpoint, best_params


@time_func
def vote_instances(args, config, data, pred_folder, output_folder):
    samples = get_list_samples(config, pred_folder,
                               config['prediction']['output_format'],
                               data)
    if config["data"].get("add_partly_val", False):
        samples += get_list_samples(
            config, pred_folder,
            config["prediction"]["output_format"],
            data.replace("complete", "partly"))

    if args.sample is not None:
        samples = [s for s in samples if args.sample in s]

    # set vote instance parameter
    config['vote_instances']['check_required'] = False
    num_workers = config['vote_instances'].get("num_parallel_samples", 1)
    if num_workers > 1:
        def init(l):
            global mutex
            mutex = l

        mutex = mp.Lock()
        pool = mp.Pool(processes=num_workers,
                       initializer=init, initargs=(mutex,))
        pool.map(functools.partial(vote_instances_sample, args, config,
                                   data, pred_folder, output_folder), samples)
        pool.close()
        pool.join()
    else:
        for idx, sample in enumerate(samples):
            print("labelling {}/{}: {}".format(idx, len(samples), sample))
            vote_instances_sample_seq(args, config, data, pred_folder,
                                      output_folder, sample)
            if args.predict_single:
                break


def cleanup(args, config, data, pred_folder, inst_folder):
    samples = get_list_samples(config, inst_folder,
                               config['vote_instances']['output_format'],
                               data)
    for sample in samples:
        vi_file = glob(os.path.join(
                inst_folder,
                sample + '*.' + config['vote_instances']['output_format']))
        pred_file = glob(os.path.join(
                pred_folder,
                sample + '*.' + config['prediction']['output_format']))
        if not pred_file:
            print("cleanup: pred does not exist, nothing to delete for sample %s" % str(sample))
        if vi_file and pred_file:
            print("cleanup: inst and pred both exist, deleting pred %s" % str(pred_file[0]))
            shutil.rmtree(pred_file[0])


# only fork if pool is not used
@fork
@time_func
def vote_instances_sample_seq(args, config, data, pred_folder, output_folder,
                              sample):
    vote_instances_sample(args, config, data, pred_folder, output_folder,
                          sample)


def vote_instances_sample(args, config, data, pred_folder, output_folder,
                          sample):
    if config['data'].get('sub_folders', False):
        config['vote_instances']['result_folder'] = \
            os.path.join(output_folder,
                         os.path.basename(os.path.dirname(sample)))
    else:
        config['vote_instances']['result_folder'] = output_folder

    if args.term_after_patch_graph:
        config['vote_instances']['termAfterPatchGraph'] = True
    if args.graph_to_inst:
        config['vote_instances']['graphToInst'] = True

    # check if instance file already exists
    output_fn = os.path.join(
        config['vote_instances']['result_folder'],
        os.path.basename(sample) + '.' + config['vote_instances']['output_format'])
    if not config['general']['overwrite'] and os.path.exists(output_fn):
        if check_file(output_fn, remove_on_error=False,
                      key=config['evaluation']['res_key']):
            logger.info('Skipping vote instances for %s. Already exists!',
                        os.path.join(
                            output_folder,
                            sample + '*.' + config['vote_instances']['output_format']))
            return
        else:
            logger.info('vote instances %s broken. recomputing..!',
                        os.path.join(
                            output_folder,
                            sample + '*.' + config['vote_instances']['output_format']))

    if config['vote_instances']['cuda'] and \
       'CUDA_VISIBLE_DEVICES' not in os.environ and \
       not config['vote_instances'].get('graphToInst', False):
        raise RuntimeError("no free GPU available!")

    if config['vote_instances']['cuda'] and \
       not config['vote_instances']['blockwise']:
        if config['vote_instances'].get("num_parallel_samples", 1) == 1:
            config['vote_instances']['mutex'] = mp.Lock()
        else:
            config['vote_instances']['mutex'] = mutex

    if config['vote_instances']['blockwise']:
        pred_file = os.path.join(
            pred_folder, sample + '.' + config['prediction']['output_format'])
        if config['vote_instances'].get('blockwise_old_stitch_fn'):
            fn = vi.ref_vote_instances_blockwise.main
        else:
            fn = vi.stitch_patch_graph.main
        fn(pred_file,
           **config['vote_instances'],
           **config['model'],
           **config['visualize'],
           aff_key=config['prediction'].get('aff_key'),
           numinst_key=config['prediction'].get('numinst_key'),
           fg_key=config['prediction'].get('fg_key'),
           fg_folder=config['prediction'].get('fg_folder'),
           fg_thresh=config['prediction'].get('fg_thresh'),
           )
    else:
        config['vote_instances']['affinities'] = os.path.join(
            pred_folder, sample + '.' + config['prediction'][
                'output_format'])
        vi.vote_instances.main(
            **config['vote_instances'],
            **config['model'],
            numinst_key=config['prediction'].get('numinst_key'),
            aff_key=config['prediction'].get('aff_key'),
            fg_key=config['prediction'].get('fg_key'),
            )


def evaluate_sample(config, args, data, sample, inst_folder, output_folder,
                    file_format):
    partly = False
    if os.path.isfile(data):
        gt_path = data
        gt_key = sample + "/gt"
    elif data.endswith(".zarr"):
        gt_path = data
        gt_key = config['data']['gt_key']
    else:
        gt_path = os.path.join(
            data, sample + "." + config['data']['input_format'])

        if config["data"].get("add_partly_val", False) and os.path.exists(
                gt_path.replace("complete", "partly")):
            gt_path = gt_path.replace("complete", "partly")
            partly = True
        gt_key = config['data']['gt_key']

    partly = "partly" in gt_path

    sample_path = os.path.join(inst_folder, sample + "." + file_format)
    if config['vote_instances'].get('one_instance_per_channel'):
        if config['data'].get('one_instance_per_channel_gt'):
            gt_key = config['data'].get('one_instance_per_channel_gt')
            logger.info('call evaluation with key %s', gt_key)

    logger.info("evaluating %s (partly? %s)", sample, partly)

    # todo: take out extra evaluate skeleton coverage
    if config['evaluation'].get('evaluate_skeleton_coverage'):
        eval_skeleton_fn = importlib.import_module(
            args.app + '.03_evaluate.evaluate').evaluate_file
        return eval_skeleton_fn(sample_path,
                                config['evaluation']['res_key'],
                                gt_path,
                                output_folder=output_folder,
                                remove_small_comps=config['evaluation'][
                                    'remove_small_comps'],
                                show_tp=config['evaluation'].get('show_tp'),
                                save_postprocessed=config['evaluation'].get(
                                    'save_postprocessed'),
                                **config["data"]
                                )

    else:
        return evaluate_file(
            sample_path, gt_path, res_key=config['evaluation']['res_key'],
            gt_key=gt_key, out_dir=output_folder, suffix="",
            foreground_only=config['evaluation'].get('foreground_only', False),
            debug=config['general']['debug'],
            from_scratch=config['evaluation'].get('from_scratch', False),
            overlapping_inst=config['vote_instances'].get(
                'one_instance_per_channel', False),
            use_linear_sum_assignment=config['evaluation'].get(
                'use_linear_sum_assignment', False),
            metric=config['evaluation'].get('metric', None),
            filterSz=config['evaluation'].get('filterSz', None),
            localization_criterion=config['evaluation'].get("localization_criterion", "iou"),
            remove_small_components=config['evaluation'].get("remove_small_components", None),
            keep_gt_shape=config['evaluation'].get("keep_gt_shape", False),
            partly=partly,
            visualize=config["evaluation"].get("visualize", False),
            visualize_type=config["evaluation"].get("visualize_type", "nuclei"),
            assignment_strategy=config["evaluation"].get("assignment_strategy", "hungarian"),
            evaluate_false_labels=config["evaluation"].get("evaluate_false_labels", False),
            unique_false_labels=config["evaluation"].get("unique_false_labels", False),
            add_general_metrics=config["evaluation"].get("add_general_metrics", []),
            add_multi_thresh_metrics=config["evaluation"].get("add_multi_thresh_metrics", []),
    )


@fork_return
@time_func
def evaluate_autoencoder(args, config, data, checkpoint,
                         train_folder, output_folder, queue):
    import tensorflow as tf

    if args.run_from_exp:
        eval_predict_fn = runpy.run_path(
            os.path.join(config['base'], 'eval_predict.py'))['run']
    else:
        eval_predict_fn = importlib.import_module(
            args.app + '.setups.' + args.setup + '.eval_predict').run

    input_shape = tuple(p for p in config['model']['train_input_shape'] if p > 1)
    checkpoint_file = get_checkpoint_file(checkpoint,
                                          config['model']['train_net_name'],
                                          train_folder)

    samples = get_list_samples(config, data, config['data']['input_format'])
    for idx, s in enumerate(samples):
        samples[idx] = os.path.join(data,
                                    s + "." + config['data']['input_format'])
    results = eval_predict_fn(mode=tf.estimator.ModeKeys.EVAL,
                              input_shape=input_shape,
                              batch_size=config['evaluation']['batch_size'],
                              max_samples=config['evaluation']['max_samples'],
                              checkpoint_file=checkpoint_file,
                              output_folder=output_folder,
                              samples=samples,
                              **config['model'],
                              **config['data'])

    queue.put(results[config['evaluation']['metric']])


def evaluate_prediction_sample(args, config, sample, data, pred_folder,
                               file_format, output_folder):
    logger.info("evaluating prediction %s", sample)
    pred_fn = os.path.join(pred_folder, sample + '.' + file_format)
    if data.endswith('zarr') or data.endswith('hdf'):
        label_fn = data
        sample_gt_key = sample + '/' + config['data']['gt_key']
    else:
        label_fn = os.path.join(data, sample + '.' + config['data']['input_format'])
        sample_gt_key = config['data']['gt_key']
    metric = {}

    if config['evaluation']['prediction'].get('eval_numinst_prediction'):
        numinst_metric = evaluate_numinst(
            pred_fn, label_fn,
            sample_gt_key=sample_gt_key,
            output_folder=output_folder,
            **config['evaluation']['prediction'],
            **config['model'],
            **config['prediction'],
            **config['data'],
            numinst_threshs=config['vote_instances'].get("numinst_threshs")
        )
        metric.update(numinst_metric)

    if config['evaluation']['prediction'].get('eval_fg_prediction'):
        fg_metric = evaluate_fg(
            pred_fn, label_fn,
            sample_gt_key=sample_gt_key,
            output_folder=output_folder,
            **config['evaluation']['prediction'],
            **config['model'],
            **config['prediction'],
            **config['data']
        )
        metric.update(fg_metric)

    if config['evaluation']['prediction'].get('eval_patch_prediction'):
        patch_metric = evaluate_patch(
            pred_fn, label_fn,
            sample_gt_key=sample_gt_key,
            **config['evaluation']['prediction'],
            **config['model'],
            **config['prediction'],
            **config['data']
        )
        metric.update(patch_metric)
        if config['evaluation']['prediction'].get('store_iou'):
            if file_format == "zarr":
                outfl = zarr.open(pred_fn, 'a')
            elif file_format == "hdf":
                outfl = h5py.File(pred_fn, 'a')
            else:
                raise RuntimeError("invalid file format")
            for th in config['evaluation']['prediction']['eval_patch_thresholds']:
                thresh_key = str(round(th, 2)).replace('.', '_')
                iou_key = "volumes/{}/IOU".format(thresh_key)
                try:
                    del outfl[iou_key]
                except KeyError:
                    pass
                outfl.create_dataset(
                    iou_key,
                    data=patch_metric[thresh_key]['rsc_0']['IOU'],
                    compression='gzip')
            if file_format == "hdf":
                outfl.close()
    return metric


def evaluate_prediction(args, config, data, pred_folder, output_folder):
    file_format = config['prediction']['output_format']
    samples = natsorted(get_list_samples(config, pred_folder, file_format, data))

    if args.sample is not None:
        samples = [s for s in samples if args.sample in s]

    num_workers = config['evaluation'].get("num_workers", 1)
    if num_workers > 1:
        metric_dicts = Parallel(n_jobs=num_workers, backend='multiprocessing',
                                verbose=0)(
            delayed(evaluate_prediction_sample)(args, config, s, data,
                                                pred_folder, file_format,
                                                output_folder)
            for s in samples)
    else:
        metric_dicts = []
        for sample in samples:
            metric = evaluate_prediction_sample(
                args, config, sample, data, pred_folder, file_format,
                output_folder)
            metric_dicts.append(metric)

    summarize_metric_dict(
        metric_dicts,
        samples,
        config['evaluation']['summary'],
        os.path.join(output_folder, 'summary_prediction.csv')
    )

    metric_dicts_reordered = {}
    for sampleMetric in metric_dicts:
        for th_key, v1 in sampleMetric.items():
            for rsc_key, v2 in v1.items():
                key = th_key + "_" + rsc_key
                if key not in metric_dicts_reordered:
                    metric_dicts_reordered[key] = []
                metric_dicts_reordered[key].append(v2)

    # # if config['evaluation']['prediction'].get('eval_numinst_prediction'):
    # return metric_dicts_reordered

    metrics = []
    ths = []
    print(metric_dicts_reordered)
    for th, metric_dicts in metric_dicts_reordered.items():
        metrics_th = []
        for metric_dict, sample in zip(metric_dicts, samples):
            if metric_dict is None:
                continue
            for k in config['evaluation']['metric'].split('.'):
                if k in metric_dict:
                    metric_dict = metric_dict[k]
            if type(metric_dict) == dict:
                logger.info("%s sample has no overlap/check metric", sample)
            else:
                logger.info("%s sample %-19s: %.4f",
                            config['evaluation']['metric'], sample,
                            float(metric_dict))
                metrics_th.append(float(metric_dict))

        metric = np.mean(metrics_th)
        metrics.append(metric)
        ths.append(th)
        logger.info("%s: %.4f (%s)",
                    config['evaluation']['metric'], metric, th)

    logger.info("%s", metrics)
    logger.info("%s", ths)
    logger.info("best %s: %.4f (%s)",
                config['evaluation']['metric'],
                np.max(metrics), ths[np.argmax(metrics)])

    return metrics, ths


@time_func
def evaluate(args, config, data, inst_folder, output_folder, return_avg=True):
    file_format = config['postprocessing']['watershed']['output_format']
    add_partly = config["data"].get("add_partly_val", False)
    if add_partly:
        samples = natsorted(get_list_samples(config, inst_folder, file_format))
        samples_complete = get_list_samples(config, data, config['data']['input_format'])
        complete = np.isin(samples, samples_complete)
    else:
        samples = natsorted(get_list_samples(config, inst_folder, file_format, data))
        complete = np.ones(len(samples), dtype=bool)

    if args.sample is not None:
        samples = [s for s in samples if args.sample in s]

    if args.app == "conic":
        eval_fn = importlib.import_module(
            args.app + '.setups.' + args.setup + '.eval').work
        metrics = eval_fn("seg_class", inst_folder, data)
        with open(os.path.join(output_folder, "eval.toml"), 'w') as f:
            toml.dump(metrics, f)
        return metrics

    num_workers = config['evaluation'].get("num_workers", 1)
    if num_workers > 1:
        metric_dicts = Parallel(n_jobs=num_workers, backend='multiprocessing',
                                verbose=0)(
            delayed(evaluate_sample)(config, args, data, s, inst_folder,
                                     output_folder, file_format)
            for s in samples)
    else:
        metric_dicts = []
        for sample in samples:
            metric_dict = evaluate_sample(config, args, data, sample,
                                          inst_folder, output_folder,
                                          file_format)
            metric_dicts.append(metric_dict)
            if args.predict_single:
                break

    metrics = {}
    metrics_full = {}
    metrics_full_partly = {}
    num_gt = 0
    num_fscore = 0
    for metric_dict, sample, cplt in zip(metric_dicts, samples, complete):
        if metric_dict is None:
            continue
        # TODO: handle partly separately?

        metrics_full[sample] = metric_dict
        if config['evaluation'].get('print_f_factor_perc_gt_0_8', False):
            num_gt += int(metric_dict['general']['Num GT'])
            num_fscore += int(
                metric_dict['confusion_matrix']['th_0_5']['Fscore_cnt'])
        # report main metric to compare different runs with each other
        # todo: should that also be possible for partly labeled data?
        for k in config['evaluation']['metric'].split('.'):
            metric_dict = metric_dict[k]
        logger.info("%s sample %-19s: %.4f",
                    config['evaluation']['metric'], sample, float(metric_dict))
        metrics[sample] = float(metric_dict)

    if add_partly and 'summary' in config['evaluation'].keys():
        summarize_metric_dict(
            metric_dicts, samples,
            config['evaluation']['summary'],
            os.path.join(output_folder, 'summary_combined.csv')
        )

    if 'summary' in config['evaluation'].keys():
        summarize_metric_dict(
            [metric_dict for metric_dict, cplt in zip(metric_dicts, complete) if cplt],
            [s for s, cplt in zip(samples, complete) if cplt],
            config['evaluation']['summary'],
            os.path.join(output_folder, 'summary.csv' if "partly" not in data else 'summary_partly.csv')
        )
    if add_partly and "summary_partly" in config["evaluation"].keys():
        summarize_metric_dict(
            [metric_dict for metric_dict, cplt in zip(metric_dicts, complete) if not cplt],
            [s for s, cplt in zip(samples, complete) if not cplt],
            config['evaluation']['summary_partly'],
            os.path.join(output_folder, 'summary_partly.csv')
        )

    if config['evaluation'].get('print_f_factor_perc_gt_0_8', False):
        logger.info("fscore (at iou0.5) percent > 0.8: %.4f", num_fscore/num_gt)
    if return_avg:
        return np.mean(list(metrics.values()))
    else:
        return metrics, metrics_full


def visualize(args, config, pred_folder, inst_folder):
    if config['visualize'].get('show_patches'):
        samples = get_list_samples(config, pred_folder,
                                   config['prediction']['output_format'])
        aff_key = config['prediction'].get('aff_key', 'volumes/pred_affs')
        for sample in config['visualize'].get('samples_to_visualize', []):
            if sample in samples:
                infn = os.path.join(
                    pred_folder,
                    sample + '.' + config['prediction']['output_format'])
                outfn = os.path.join(
                    pred_folder,
                    sample + '.hdf')
                out_key = aff_key + '_patched'
                if not os.path.exists(outfn):
                    _ = visualize_patches(infn, config['model']['patchshape'],
                                          in_key=aff_key, out_file=outfn,
                                          out_key=out_key)
    if config['visualize'].get('show_instances'):
        param_sets = list(named_params(
        *get_postprocessing_params(
            config['validation'],
            config['validation'].get(
                'params_product',
                config['validation'].get('params', [])),
            config['validation'].get('params_zip', []),
            config['vote_instances']
        )))
        print(param_sets)
        for param_set in param_sets:
            params_str = [k + "_" + str(v).replace(".", "_").replace(
                " ", "").replace(",", "_").replace("[", "_").replace(
                    "]", "_").replace("(", "_").replace(")", "_")
                          for k, v in params.items()]
            vis_config = deepcopy(config)
            for k in param_set.keys():
                if "filterSz" in k or "res_key" in k:
                    vis_config['evaluation'][k] = param_set[k]
                else:
                    vis_config['vote_instances'][k] = param_set[k]

            current_inst_folder = os.path.join(inst_folder, *params_str)
            if not os.path.exists(current_inst_folder):
                continue
            samples = get_list_samples(
                vis_config, current_inst_folder,
                vis_config['vote_instances']['output_format'])
            inst_key = 'vote_instances'

            if 'one_instance_per_channel' in param_set:
                max_axis = 0 if param_set['one_instance_per_channel'] else None
            else:
                if vis_config['vote_instances'].get('one_instance_per_channel'):
                    max_axis = 0
                else:
                    max_axis = None

            for sample in vis_config['visualize'].get('samples_to_visualize', []):
                if sample in samples:
                    infn = os.path.join(
                        current_inst_folder,
                        sample + '.' + vis_config['vote_instances']['output_format'])
                    outfn = os.path.join(
                        current_inst_folder,
                        sample + '.png')
                    if not os.path.exists(outfn):
                        visualize_instances(infn, inst_key, outfn,
                                            max_axis=max_axis)


def average_flylight_score_over_instances(samples_foldn, result):
    # heads up: hard coded for average F1 * average gt coverage
    threshs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    fscores = []
    gt_covs = []
    tp = {}
    fp = {}
    fn = {}
    false_split = []
    false_merge = []
    for thresh in threshs:
        tp[thresh] = []
        fp[thresh] = []
        fn[thresh] = []
    for s in samples_foldn:
        # todo: move type conversion to evaluate_file
        gt_covs += list(np.array(
            result[1][s]["general"]["gt_skel_coverage"], dtype=np.float32))
        for thresh in threshs:
            tp[thresh].append(result[1][s][
                "confusion_matrix"]["th_" + str(thresh).replace(".","_")]["AP_TP"])
            fp[thresh].append(result[1][s][
                    "confusion_matrix"]["th_" + str(thresh).replace(".","_")]["AP_FP"])
            fn[thresh].append(result[1][s][
                    "confusion_matrix"]["th_" + str(thresh).replace(".","_")]["AP_FN"])
            if thresh == 0.5:
                false_split.append(result[1][s][
                    "confusion_matrix"]["th_0_5"]["false_split"])
                false_merge.append(result[1][s][
                    "confusion_matrix"]["th_0_5"]["false_merge"])
    for thresh in threshs:
        fscores.append(2*np.sum(tp[thresh]) / (
            2*np.sum(tp[thresh]) + np.sum(fp[thresh]) + np.sum(fn[thresh])))
    avS = 0.5 * np.mean(fscores) + 0.5 * np.mean(gt_covs)
    per_instance_counts = {}
    per_instance_counts["gt_covs"] = gt_covs
    per_instance_counts["false_split"] = np.sum(false_split)
    per_instance_counts["false_merge"] = np.sum(false_merge)
    per_instance_counts["tp"] = []
    per_instance_counts["fp"] = []
    per_instance_counts["fn"] = []
    for thresh in threshs:
        per_instance_counts["tp"].append(np.sum(tp[thresh]))
        per_instance_counts["fp"].append(np.sum(fp[thresh]))
        per_instance_counts["fn"].append(np.sum(fn[thresh]))
    return avS, per_instance_counts


def average_flylight_score_with_instance_counts(results_fold1, results_fold2):
    gt_covs = results_fold1["gt_covs"] + results_fold2["gt_covs"]
    tps = np.array(results_fold1["tp"]) + np.array(results_fold2["tp"])
    fps = np.array(results_fold1["fp"]) + np.array(results_fold2["fp"])
    fns = np.array(results_fold1["fn"]) + np.array(results_fold2["fn"])
    fscores = 2 * tps / (2 * tps + fps + fns)
    avg_f1_cov_score = 0.5 * np.mean(gt_covs) + 0.5 * np.mean(fscores)
    acc_both_folds = {}
    acc_both_folds["avg_f1_cov_score"] = avg_f1_cov_score
    acc_both_folds["avg_gt_skel_coverage"] = np.mean(gt_covs)
    acc_both_folds["fscores"] = fscores
    acc_both_folds["avFscore"] = np.mean(fscores)
    acc_both_folds["false_split"] = results_fold1["false_split"] + results_fold2["false_split"]
    acc_both_folds["false_merge"] = results_fold1["false_merge"] + results_fold2["false_merge"]
    return avg_f1_cov_score, acc_both_folds


@time_func
def cross_validate(args, config, data, checkpoints, train_folder, test_folder):
    # do cross validation either on all samples in one folder or two separate folders
    if config["data"].get("cross_val_folders", None) is not None:
        nfolds = []
        for foldi in config["data"].get("cross_val_folders"):
            nfolds.append(config["data"][foldi])
        print(nfolds)
    else:
        nfolds = [data]
    # run predict, vote instances and evaluate for all samples
    param_sets = list(named_params(
        *get_postprocessing_params(
            config['validation'],
            config['validation'].get(
                'params_product',
                config['validation'].get('params', [])),
            config['validation'].get('params_zip', []),
            config['vote_instances']
        )))
    logger.info("val params %s", param_sets)

    for checkpoint in checkpoints:
        logger.info("cross-validating checkpoint %s", checkpoint)
        for foldi in nfolds:
            logger.info("cross-validating fold %s", foldi)
            pred_folder = os.path.join(test_folder, 'processed', str(checkpoint))
            os.makedirs(pred_folder, exist_ok=True)
            checkpoint_file = get_checkpoint_file(checkpoint,
                                                  config['model']['train_net_name'],
                                                  train_folder)
            if not args.skip_predict:
                predict(args, config, config['model']['test_net_name'], foldi,
                        checkpoint_file, test_folder, pred_folder)
            # if ppp learns code
            if config['training'].get('train_code') or \
               config['model'].get('train_code'):
                autoencoder_chkpt = config['model']['autoencoder_chkpt']
                if autoencoder_chkpt == "this":
                    autoencoder_chkpt = checkpoint_file
                decode(args, config, foldi, autoencoder_chkpt, pred_folder,
                       pred_folder)

    if args.only_predict_decode:
        return

    results = {}
    # results_numinst = {}
    # results_patch = {}
    for checkpoint in checkpoints:
        logger.info("cross-validating checkpoint %s", checkpoint)
        for idx, param_set in enumerate(param_sets):
            if args.val_id >= 0 and args.val_id != idx:
                continue

            params_str = [k + "_" + str(v).replace(".", "_").replace(
                " ", "").replace(",", "_").replace("[", "_").replace(
                    "]", "_").replace("(", "_").replace(")", "_")
                          for k, v in param_set.items()]
            param_set = {
                k:v if not isinstance(v, list) else tuple(v)
                for k, v in param_set.items()}
            # change values in config
            print("params_set: ", param_set)
            print("params_str:", params_str)
            val_config = deepcopy(config)
            for k in param_set.keys():
                val_config['vote_instances'][k] = param_set[k]
            inst_folder = os.path.join(
                test_folder, 'instanced', str(checkpoint), *params_str)
            eval_folder = os.path.join(
                test_folder, 'evaluated', str(checkpoint), *params_str)
            os.makedirs(inst_folder, exist_ok=True)
            os.makedirs(eval_folder, exist_ok=True)
            logger.info("vote instances: %s", param_set)

            print('call vi with ', pred_folder, inst_folder, param_set)
            #results[(checkpoint, *(param_set.values()))] = [] # todo: check return value evaluate
            for foldi in nfolds:
                # if not args.skip_evaluate:
                    vote_instances(args, val_config, foldi, pred_folder, inst_folder)
                # if val_config['evaluation'].get('prediction'):
                #     if val_config['evaluation']['prediction'].get('eval_numinst_prediction'):
                #         val_config_ep = deepcopy(val_config)
                #         val_config_ep['evaluation']['prediction'] = {
                #             'eval_numinst_prediction': True}
                #         if tuple([foldi, param_set['numinst_threshs']]) not in results_numinst:
                #             t = evaluate_prediction(
                #                 args, val_config_ep, foldi, pred_folder,
                #                 os.path.join(test_folder, "evaluated", str(checkpoint)))
                #             results_numinst[(foldi, param_set['numinst_threshs'])] = t['1_f1']

                #     if val_config['evaluation']['prediction'].get('eval_patch_prediction'):
                #         val_config_ep = deepcopy(val_config)
                #         val_config_ep['evaluation']['prediction'] = {
                #             'eval_patch_prediction': True,
                #             'eval_patch_thresholds': val_config['evaluation']['prediction']['eval_patch_thresholds']}
                #         if tuple([foldi]) not in results_patch:
                #             t = evaluate_prediction(
                #                 args, val_config_ep, foldi, pred_folder,
                #                 os.path.join(test_folder, "evaluated", str(checkpoint)))
                #             results_patch[(foldi,)] = t

                # if not args.skip_evaluate:
                    metrics = evaluate(
                        args, val_config, foldi, inst_folder, eval_folder,
                        return_avg=False)
                    if (checkpoint, *(param_set.values())) not in results:
                        results[(checkpoint, *(param_set.values()))] = metrics
                    else:
                        results[(checkpoint, *(param_set.values()))][0].update(metrics[0])
                        results[(checkpoint, *(param_set.values()))][1].update(metrics[1])

    if args.sample is not None:
        return

    # perform two-fold cross validation on processed samples
    if len(nfolds)==1:
        samples = get_list_samples(config, data, config['data']['input_format'])
    else:
        samples_fold1 = get_list_samples(config, nfolds[0], config['data']['input_format'])
        samples_fold2 = get_list_samples(config, nfolds[1], config['data']['input_format'])
        samples = samples_fold1 + samples_fold2
    #     # results_numinst_t = {}
    #     # for ((f, nth), res) in results_numinst.items():
    #     #     results_numinst_t.setdefault(nth, []).append(res)
    #     # results_numinst = results_numinst_t
    #     results_patch_t = {}
    #     print(results_patch)
    #     for ((f,), res_t) in results_patch.items():
    #         for (k, res_tt) in res_t.items():
    #             if k not in results_patch_t:
    #                 results_patch_t[k] = {
    #                     "precision": [],
    #                     "recall": [],
    #                     "f1": [],
    #                 }
    #             for res in res_tt:
    #                 results_patch_t[k]['precision'].append(res['precision'])
    #                 results_patch_t[k]['recall'].append(res['recall'])
    #                 results_patch_t[k]['f1'].append(res['f1'])
    #     results_patch = results_patch_t
    #     print(results_patch)
    # logger.info("numinst")
    # for (nth, res) in results_numinst.items():
    #     logger.info("%s %s", nth, np.mean(res))
    # logger.info("patch")
    # for k in ['precision', 'recall', 'f1']:
    #     for (nth, met) in results_patch.items():
    #     # for k, v in met.items():
    #         v = met[k]
    #         logger.info("%s %s %s", nth, k, np.mean(v))

    print(samples)
    samples = natsorted(samples)
    # check if there is a result for each sample
    for k, v in results.items():
        assert len(v[0]) == len(samples)
        for s1, s2 in zip(natsorted(v[0].keys()), samples):
            assert s1 == s2
    # split samples of one set
    if len(nfolds) == 1:
        random.Random(42).shuffle(samples)
        samples_fold1 = set(samples[:len(samples)//2])
        samples_fold2 = set(samples[len(samples)//2:])
        logger.info("Shuffling with random.Random(42).shuffle(samples)")
        logger.info("Samples in fold 1: %s", sorted(samples_fold1))
        logger.info("Samples in fold 2: %s", sorted(samples_fold2))
    # or take samples from two folders
    else:
        samples_fold1 = set(samples_fold1)
        samples_fold2 = set(samples_fold2)

    # average results either over samples or instances for each setup
    results_fold1 = {}
    results_fold2 = {}
    results_fold1_complete = {}
    results_fold2_complete = {}
    if config["evaluation"].get("average_over_samples", True) == True:
        for setup, result in results.items():
            acc = []
            for s in samples_fold1:
                acc.append(result[0][s])
            acc = np.mean(acc)
            results_fold1[setup] = acc

            acc = []
            for s in samples_fold2:
                acc.append(result[0][s])
            acc = np.mean(acc)
            results_fold2[setup] = acc
    else:
        if config["evaluation"].get("metric", None) == "general.avg_f1_cov_score":
            for setup, result in results.items():
                # average over instances for fold1
                acc, acc_per_instance = average_flylight_score_over_instances(
                        samples_fold1, result)
                results_fold1[setup] = acc
                results_fold1_complete[setup] = acc_per_instance.copy()
                # average over instances for fold2
                acc, acc_per_instance = average_flylight_score_over_instances(
                        samples_fold2, result)
                results_fold2[setup] = acc
                results_fold2_complete[setup] = acc_per_instance.copy()
        else:
            raise NotImplementedError
    for setup in results_fold1.keys():
        acc, acc_all = average_flylight_score_with_instance_counts(
            results_fold1_complete[setup],
            results_fold2_complete[setup]
        )
        logger.info("%s VAL %s: %.4f",
                    config['evaluation']['metric'], setup, acc)

    # select best setup for each fold
    best_setup_fold1 = max(results_fold1.items(), key=operator.itemgetter(1))[0]
    best_setup_fold2 = max(results_fold2.items(), key=operator.itemgetter(1))[0]

    # todo: check with Peter: why averaging again here?
    if config["evaluation"].get("average_over_samples", True) == True:
        acc_fold2 = []
        for s in samples_fold2:
            acc_fold2.append(results[best_setup_fold1][0][s])
        acc_fold1 = []
        for s in samples_fold1:
            acc_fold1.append(results[best_setup_fold2][0][s])

        acc = np.mean(acc_fold2 + acc_fold1)
        acc_fold2 = np.mean(acc_fold2)
        acc_fold1 = np.mean(acc_fold1)
    else:
        acc_fold1 = results_fold1[best_setup_fold2]
        acc_fold2 = results_fold2[best_setup_fold1]
        acc, acc_all = average_flylight_score_with_instance_counts(
                results_fold1_complete[best_setup_fold2],
                results_fold2_complete[best_setup_fold1]
                )

    logger.info("%s CROSS: %.4f [%.4f (%s), %.4f (%s)]",
                config['evaluation']['metric'], acc,
                acc_fold1, best_setup_fold2,
                acc_fold2, best_setup_fold1)
    print("%s CROSS: %.4f [%.4f (%s), %.4f (%s)]" % (
        config['evaluation']['metric'], acc,
        acc_fold1, best_setup_fold2,
        acc_fold2, best_setup_fold1))

    if config["evaluation"].get("average_over_samples", True) == True:
        ap_ths = ["confusion_matrix.avAP",
                  "confusion_matrix.th_0_5.AP",
                  "confusion_matrix.th_0_6.AP",
                  "confusion_matrix.th_0_7.AP",
                  "confusion_matrix.th_0_75.AP",
                  "confusion_matrix.th_0_8.AP",
                  "confusion_matrix.th_0_9.AP",
                  "confusion_matrix.th_0_5.recall",
                  "confusion_matrix.th_0_8.recall",
                  ]
        for ap_th in ap_ths:
            metrics = {}
            # get AP's for fold1
            metric_dicts = results[best_setup_fold2][1]
            for sample, metric_dict in metric_dicts.items():
                if sample not in samples_fold1:
                   continue
                if metric_dict is None:
                    continue
                for k in ap_th.split('.'):
                    metric_dict = metric_dict[k]
                metrics[sample] = float(metric_dict)

            # get AP's for fold2
            metric_dicts = results[best_setup_fold1][1]
            for sample, metric_dict in metric_dicts.items():
                if sample not in samples_fold2:
                    continue
                if metric_dict is None:
                    continue
                for k in ap_th.split('.'):
                    metric_dict = metric_dict[k]
                metrics[sample] = float(metric_dict)
            metric = np.mean(list(metrics.values()))
            logger.info("%s: %.4f", ap_th, metric)
    else:
        # write summary for cross validation result
        out_folder = os.path.join(test_folder, 'evaluated')
        # write cross validation params
        cv_info = {}
        cv_info["data"] = nfolds
        cv_info["samples_fold1"] = samples_fold1
        cv_info["samples_fold2"] = samples_fold2
        cv_info["setups"] = list(results.keys())
        cv_info["best_setup_fold1"] = best_setup_fold1
        cv_info["best_setup_fold2"] = best_setup_fold2
        cv_info["average_over_samples"] = config["evaluation"].get("average_over_samples", True)
        toml_fn = open(os.path.join(out_folder, "cross_validate_info.toml"), "w")
        toml.dump(cv_info, toml_fn)
        print(acc_all)


def main():
    # parse command line arguments
    args = get_arguments()

    if not args.do:
        raise ValueError("Provide a task to do (-d/--do)")

    # get experiment name
    is_new_run = True
    if args.expid is not None:
        if os.path.isdir(args.expid):
            base = args.expid
            is_new_run = False
        else:
            base = os.path.join(args.root, args.expid)
    else:
        base = os.path.join(args.root,
                            args.app + '_' + args.setup + '_' + \
                            datetime.now().strftime('%y%m%d_%H%M%S_%f'))

    # create folder structure for experiment
    if args.debug_args:
        base = base.replace("experiments", "experimentsTmp")
    train_folder, val_folder, test_folder = create_folders(args, base)

    # read config file
    if args.config is None and args.expid is None:
        raise RuntimeError("No config file provided (-c/--config)")
    elif args.config is None:
        args.config = [os.path.join(base, 'config.toml')]
    try:
        config = {}
        for conf in args.config:
            config = merge_dicts(config, toml.load(conf))
    except:
        raise IOError('Could not read config file: {}! Please check!'.format(
            conf))
    config['base'] = base
    os.makedirs(os.path.join(base, "backups"), exist_ok=True)

    repo = Repo(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
    diff = repo.git.diff("HEAD")
    dttm = datetime.now().strftime('%y%m%d_%H%M%S')
    if diff is not None:
        with open(os.path.join(
                base, "backups",
                "ppp_source_" + dttm +".diff"), "w"
        ) as f:
            f.write(diff)

    # set logging level
    prefix = "" if args.sample is None else args.sample
    logging.basicConfig(
        level=config['general']['logging'],
        handlers=[
            logging.FileHandler(os.path.join(base,
                                             prefix + "run.log"), mode='a'),
            logging.StreamHandler(sys.stdout)
        ])
    if args.comment is not None:
        logger.info("note: %s", args.comment)
        print("note: ", args.comment)
    logger.info('attention: using config file %s', args.config)

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        try:
            selectedGPU = util.selectGPU(
                quantity=config['training']['num_gpus'])
        except FileNotFoundError:
            selectedGPU = None
        if selectedGPU is None:
            logger.warning("no free GPU available!")
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                [str(i) for i in selectedGPU])
        logger.info("setting CUDA_VISIBLE_DEVICES to device {}".format(
            selectedGPU))
    else:
        logger.info("CUDA_VISIBILE_DEVICES already set, device {}".format(
            os.environ["CUDA_VISIBLE_DEVICES"]))

    # hpc race condition temp fix
    try:
        os.path.isdir(config['data']['train_data'])
    except:
        pass

    # update config with command line values
    update_config(args, config)
    if is_new_run:
        with open(os.path.join(
                base, f"config.toml"), 'w') as f:
            toml.dump(config, f)
    else:
        with open(os.path.join(
                base, "backups", f"config_backup_{dttm}.toml"), 'w') as f:
            toml.dump(config, f)
    if args.debug_args:
        setDebugValuesForConfig(config)
    logger.info('used config: %s', config)

    # create network
    if 'all' in args.do or 'mknet' in args.do:
        mknet(args, config, train_folder, test_folder)

    # train network
    if 'all' in args.do or 'train' in args.do:
        train(args, config, train_folder)

    # determine which checkpoint to use
    checkpoint = None
    if any(i in args.do for i in ['all', 'validate', 'predict', 'decode',
                                  'label', 'infer', 'evaluate', 'visualize',
                                  'postprocess', 'cleanup']):
        if args.checkpoint is not None:
            checkpoint = int(args.checkpoint)
            checkpoint_path = os.path.join(
                train_folder, config['model']['train_net_name'], '_checkpoint_',
                str(checkpoint))

        elif args.test_checkpoint == 'last':
            checkpoint = util.get_latest_checkpoint(
                os.path.join(
                    train_folder, config['model']['train_net_name']))[1]
            if checkpoint == 0:
                checkpoint = None

        if checkpoint is None and \
           args.test_checkpoint != 'best' and \
           any(i in args.do for i in ['validate', 'predict', 'decode',
                                      'label', 'evaluate', 'infer']):
            raise ValueError(
                'Please provide a checkpoint (--checkpoint/--test-checkpoint)')

    params = None
    if args.checkpoint is not None:
        checkpoints = [int(args.checkpoint)]
    elif config['validation'].get('checkpoints'):
        checkpoints = config['validation']['checkpoints']
    else:
        checkpoints = get_checkpoint_list(config['model']['train_net_name'],
                                          train_folder)
    # validation:
    # validate all checkpoints
    if ([do for do in args.do if do in ['all', 'predict', 'label',
                                       'postprocess', 'evaluate']]\
        and args.test_checkpoint == 'best') \
        or 'validate_checkpoints' in args.do:
        data, output_folder = select_validation_data(config, train_folder,
                                                     val_folder)

        logger.info("validating all checkpoints")
        checkpoint, params = validate_checkpoints(args, config, data,
                                                  checkpoints,
                                                  train_folder, test_folder,
                                                  output_folder)
    # validate single checkpoint
    else:
        if 'validate' in args.do:
            if checkpoint is None:
                raise RuntimeError("checkpoint must be set but is None")
            data, output_folder = select_validation_data(config, train_folder,
                                                         val_folder)
            _ = validate_checkpoints(args, config, data, [checkpoint],
                                     train_folder, test_folder, output_folder)

    if [do for do in args.do if do in ['all', 'predict', 'decode', 'label',
                                       'postprocess', 'evaluate', 'infer']]:
        if checkpoint is None:
            raise RuntimeError("checkpoint must be set but is None")
        checkpoint_file = get_checkpoint_file(
            checkpoint, config['model']['train_net_name'], train_folder)

        if args.app == 'autoencoder':
            params = {}
        elif args.test_checkpoint != 'best':
            params = merge_dicts(*get_postprocessing_params(
                None,
                config['validation'].get(
                    'params_product',
                    config['validation'].get('params', [])),
                config['validation'].get('params_zip', []),
                config['vote_instances']
            ))
            for k,v in params.items():
                params[k] = v[0]
        else:
            # update config with "best" params
            for k in list(params.keys()):
                if "filterSz" in k or "res_key" in k:
                    config['evaluation'][k] = params[k]
                    del params[k]
                else:
                    config['vote_instances'][k] = params[k]

        params_str = [k + "_" + str(v).replace(".", "_").replace(
            " ", "").replace(",", "_").replace("[", "_").replace(
                "]", "_").replace("(", "_").replace(")", "_")
                      for k, v in params.items()]
        pred_folder = os.path.join(test_folder, 'processed', str(checkpoint))
        inst_folder = os.path.join(test_folder, 'instanced', str(checkpoint),
                                   *params_str)
        eval_folder = os.path.join(test_folder, 'evaluated', str(checkpoint),
                                   *params_str)

    # predict test set
    if ('all' in args.do or 'predict' in args.do or 'infer' in args.do) and \
       not args.skip_predict:

        # assume checkpoint has been determined already
        os.makedirs(pred_folder, exist_ok=True)

        logger.info("predicting checkpoint %d", checkpoint)
        if args.app == "autoencoder":
            predict_autoencoder(args, config, config['data']['test_data'],
                                checkpoint, train_folder, pred_folder)
        else:
            predict(args, config, config['model']['test_net_name'],
                    config['data']['test_data'], checkpoint_file,
                    test_folder, pred_folder)

    if 'all' in args.do or 'decode' in args.do or 'infer' in args.do:
        if config['training'].get('train_code') or \
           config['model'].get('train_code'):
            autoencoder_chkpt = config['model'].get('autoencoder_chkpt')
            if autoencoder_chkpt == "this":
                autoencoder_chkpt = checkpoint_file
            decode(args, config, config['data']['test_data'],
                   autoencoder_chkpt, pred_folder, pred_folder)
        elif 'decode' in args.do:
            raise RuntimeError("Asked to decode but train_code is not set!")

    if config['evaluation'].get('prediction_only_test'):
        logger.info("evaluating prediction checkpoint %d", checkpoint)
        return evaluate_prediction(
            args, config, config['data']['test_data'], pred_folder,
            os.path.join(test_folder, "evaluated", str(checkpoint)))

    if 'all' in args.do or 'label' in args.do or 'infer' in args.do:
        os.makedirs(inst_folder, exist_ok=True)
        logger.info("vote_instances checkpoint %d", checkpoint)
        vote_instances(args, config, config['data']['test_data'], pred_folder,
                       inst_folder)

        if config['data']['test_data'].endswith("npy"):
            samples = natsorted(os.listdir(inst_folder))
            samples_data = []
            for sample in samples:
                print("loading", sample)
                with h5py.File(os.path.join(inst_folder, sample)) as f:
                    sample = np.array(f['vote_instances'])
                    samples_data.append(sample)
            samples_data = np.vstack(samples_data)
            np.save("pred_inst_ppp.npy", samples_data)

    if 'all' in args.do or 'postprocess' in args.do:
        # remove small components should go here?
        # what else?
        print('postprocess')
        if config['postprocessing'].get('process_fg_prediction', False):
            os.makedirs(inst_folder, exist_ok=True)
            samples = get_list_samples(config, pred_folder,
                                       config['prediction']['output_format'])
            samples = [os.path.join(pred_folder, s + '.' + config['prediction'][
                'output_format']) for s in samples]
            logger.info("postprocess checkpoint %d", checkpoint)
            util.postprocess_fg(
                samples, inst_folder,
                fg_key=config['prediction']['fg_key'],
                **config['postprocessing']
            )
        if config['postprocessing'].get('process_instances', False):
            samples = get_list_samples(
                config, inst_folder,
                config['vote_instances']['output_format'],
                config['data']['test_data']
            )
            samples = [os.path.join(inst_folder, s + '.' + config[
                'vote_instances']['output_format']) for s in samples]
            util.postprocess_instances(
                samples,
                inst_folder,
                res_key=config['evaluation']['res_key'],
                **config['postprocessing']
            )

    if 'all' in args.do or 'evaluate' in args.do or 'infer' in args.do:
        os.makedirs(eval_folder, exist_ok=True)
        logger.info("evaluating checkpoint %d", checkpoint)
        if args.app == "autoencoder":
            metric = evaluate_autoencoder(args, config,
                                          config['data']['test_data'],
                                          checkpoint,
                                          train_folder, eval_folder)
        else:
            metric = evaluate(args, config, config['data']['test_data'],
                              inst_folder, eval_folder)
        logger.info("%s TEST checkpoint %d: " + ("%s" if isinstance(metric, dict) else "%.4f") + " (%s)",
                    config['evaluation']['metric'], checkpoint, metric,
                    params)

    if 'visualize' in args.do:
        if checkpoint is None:
            raise RuntimeError("checkpoint must be set but is None")
        pred_folder = os.path.join(val_folder, 'processed', str(checkpoint))
        inst_folder = os.path.join(val_folder, 'instanced', str(checkpoint))

        visualize(args, config, pred_folder, inst_folder)

    if 'cross_validate' in args.do:
        cross_validate(
            args, config, config['data']['val_data'], checkpoints,
            train_folder, val_folder)

    if 'cleanup' in args.do:
        # if result from 'label' exists, dump result from 'predict'
        pred_folder = os.path.join(test_folder, str(checkpoint), 'processed')
        inst_folder = os.path.join(test_folder, str(checkpoint), 'instanced')
        cleanup(args, config, config['data']['test_data'], pred_folder, inst_folder)


if __name__ == "__main__":
    main()
