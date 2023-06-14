"""Script for training process

Create model and train
"""
import logging
import os
import time

import h5py
import numpy as np
import scipy.special
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import zarr
import random

import gunpowder as gp
import neurolight.gunpowder as nl

from . import torch_model
from . import torch_loss
from PatchPerPix.util import (
    get_latest_checkpoint,
    normalize)

logger = logging.getLogger(__name__)


def train_until(**config):
    """Main train function

    All information is taken from config object (what model architecture,
    optimizer, loss, data, augmentation etc to use)
    Train for config.train.max_iterations steps.
    Optionally compute interleaved validation statistics.

    Args
    ----
    config
    """
    # Get the latest checkpoint
    checkpoint_basename = os.path.join(config['output_folder'], 'train_net')
    _, trained_until = get_latest_checkpoint(checkpoint_basename)
    # training already done?
    if trained_until >= config["max_iterations"]:
        logger.info(
            "Model has already been trained for %s iterations", trained_until)
        return

    # check for validation
    if not 'val_log_step' in config:
        config['val_log_step'] = None

    add_affinities = config.get("add_affinities", "cpu")
    assert add_affinities in ["cpu", "torch", "loss"], (
        "Please set add_affinities to valid value!")

    raw = gp.ArrayKey('RAW')
    # raw_cropped = gp.ArrayKey('RAW_CROPPED')
    gt_instances = gp.ArrayKey('GT_INSTANCES')
    gt_labels = gp.ArrayKey('GT_LABELS')
    if add_affinities in ["cpu", "torch"]:
        gt_affs = gp.ArrayKey('GT_AFFS')
    sample_fg_points = config.get("sample_fg_points", False)
    if sample_fg_points:
        gt_fg_points = gp.GraphKey('GT_FG_POINTS')
    else:
        gt_sample_mask = gp.ArrayKey('GT_SAMPLE_MASK')
    gt_close_to_overlap = gp.GraphKey('GT_CLOSE_TO_OVERLAP')

    train_code = config.get("train_code")
    if train_code:
        pred_code = gp.ArrayKey('PRED_CODE')
        pred_code_gradients = gp.ArrayKey('PRED_CODE_GRADIENTS')

    gt_affs_samples = gp.ArrayKey('GT_AFFS_SAMPLES')
    pred_affs = gp.ArrayKey('PRED_AFFS')
    pred_affs_gradients = gp.ArrayKey('PRED_AFFS_GRADIENTS')

    overlapping_inst = config.get('overlapping_inst')
    if overlapping_inst:
        pred_numinst = gp.ArrayKey('PRED_NUMINST')
        gt_numinst = gp.ArrayKey('GT_NUMINST')
    else:
        pred_fgbg = gp.ArrayKey('PRED_FGBG')
    gt_fgbg = gp.ArrayKey('GT_FGBG')
    gt_fgbg_loss = gp.ArrayKey('GT_FGBG_LOSS')

    add_partly = config.get('add_partly', False)
    if add_partly:
        gt_fgbg_extra = gp.ArrayKey('GT_FGBG_EXTRA')
        loss_mask = gp.ArrayKey('LOSS_MASK')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("GPU: %s",
                torch.cuda.get_device_name(torch.cuda.current_device()))
    torch.backends.cudnn.benchmark = True

    model = torch_model.UnetModelWrapper(config, device, trained_until)
    model.init_layers()
    try:
        model = model.to(device)
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to move model to device. If you are using a child process "
            "to run your model, maybe you already initialized CUDA by sending "
            "your model to device in the main process."
        ) from e

    if config.get("network_style", "unet").lower() == "unet":
        model.set_padding(config['val_padding'])
        if config['val_padding'] == 'valid':
            test_input_shape = config["test_input_shape_valid"]
        else:
            test_input_shape = config["test_input_shape_same"]
        input_shape_val, output_shape_val = model.inout_shapes(
            test_input_shape, "test_net", training=False)
        print("input/output shape val: ", input_shape_val, output_shape_val)

        model.set_padding(config['train_padding'])
        if config['train_padding'] == 'valid':
            train_input_shape = config["train_input_shape_valid"]
        else:
            train_input_shape = config["train_input_shape_same"]
        input_shape, output_shape = model.inout_shapes(
            train_input_shape, "train_net", training=True)
        print("input/output shape: ", input_shape, output_shape)
    else:
        input_shape = config["train_input_shape_same"]
        output_shape = config["train_input_shape_same"]
        input_shape_val = config["test_input_shape_same"]
        output_shape_val = config["test_input_shape_same"]

    logger.debug("Model: %s", model)
    model.train()

    voxel_size = gp.Coordinate(config["voxel_size"])
    is_3d = len(voxel_size) == 3
    if not is_3d:
        assert config['patchshape'][0] == 1, (
            " if 2d data is used, please supply patchshape with 3 values and "
            "a 1 as the first dimension")

    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    # define request
    request = gp.BatchRequest()
    request.add(raw, input_size)
    if add_affinities in ["cpu", "torch"]:
        request.add(gt_affs, output_size)
    if overlapping_inst:
        if add_affinities == "loss":
            request.add(gt_instances, gp.Coordinate(
                (o+p-1 for o, p in zip(
                    output_size, config['patchshape'][-len(voxel_size):]))))
        else:
            request.add(gt_instances, output_size)
        request.add(gt_numinst, output_size)
    else:
        request.add(gt_labels, output_size)
        request.add(gt_fgbg, output_size)
    if add_partly:
        request.add(loss_mask, output_size)
    logger.debug("REQUEST: %s" % str(request))

    # define validation request
    val_input_size = gp.Coordinate(input_shape_val) * voxel_size
    val_output_size = gp.Coordinate(output_shape_val) * voxel_size
    val_request = None
    val_request = gp.BatchRequest()
    val_request.add(raw, val_input_size)
    if add_affinities in ["cpu", "torch"]:
        val_request.add(gt_affs, val_output_size)
    if overlapping_inst:
        if add_affinities == "loss":
            val_request.add(gt_instances, gp.Coordinate(
                (o+p-1 for o, p in zip(
                    val_output_size, config['patchshape'][-len(voxel_size):]))))
        else:
            val_request.add(gt_instances, val_output_size)
        val_request.add(gt_numinst, val_output_size)
    else:
        val_request.add(gt_labels, val_output_size)
        val_request.add(gt_fgbg, val_output_size)
    if add_partly:
        val_request.add(loss_mask, val_output_size)
    logger.debug("Validation request: %s" % str(val_request))

    # define snapshot request
    snapshot_request = gp.BatchRequest()
    snapshot_request.add(raw, input_size)
    if train_code:
        snapshot_request.add(pred_code, output_size)
        snapshot_request[gt_affs_samples] =  gp.ArraySpec(
            nonspatial=True)
        snapshot_request[pred_affs] =  gp.ArraySpec(
            nonspatial=True)
    else:
        snapshot_request.add(pred_affs, output_size)
        snapshot_request.add(gt_affs_samples, output_size)

    if overlapping_inst:
        snapshot_request.add(pred_numinst, output_size)
        snapshot_request.add(gt_numinst, output_size)
    else:
        snapshot_request.add(pred_fgbg, output_size)
        snapshot_request.add(gt_fgbg, output_size)
    logger.debug("Snapshot request: %s" % str(snapshot_request))
    
    # get train sources
    arrays = {
        'raw': raw,
        }
    graphs = {
            'gt_close_to_overlap': gt_close_to_overlap
            }
    if overlapping_inst:
        arrays['gt_numinst'] = gt_numinst
        arrays['gt_instances'] = gt_instances
    else:
        arrays['gt_fgbg'] = gt_fgbg
        arrays['gt_labels'] = gt_labels
    if add_partly:
        arrays['gt_fgbg_extra'] = gt_fgbg_extra
        arrays['loss_mask'] = loss_mask
    if sample_fg_points:
        graphs['gt_fg_points'] = gt_fg_points
    else:
        arrays['gt_sample_mask'] = gt_sample_mask

    train_sources, gt_max_num_inst = get_sources(
        config, arrays, voxel_size, config['data_files'], val=False, graphs=graphs)

    # Do interleaved validation?
    if config["val_log_step"] is not None:
        val_sources, gt_max_num_inst_val = get_sources(
            config, arrays, voxel_size, config['val_files'],
            val=True, graphs=graphs,
            )

    neighborhood = []
    psH = np.array(config['patchshape'])//2
    for k in range(-psH[0], psH[0]+1, config['patchstride'][0]):
        for i in range(-psH[1], psH[1]+1, config['patchstride'][1]):
            for j in range(-psH[2], psH[2]+1, config['patchstride'][2]):
                if is_3d:
                    neighborhood.append([k, i, j])
                else:
                    neighborhood.append([i,j])

    # set up pipeline:
    # load data source and
    # choose augmentations depending on config
    augment = config.get("augmentation", {})
    train_pipeline = (
        train_sources +

        (gp.ElasticAugment(
            augment["elastic"]["control_point_spacing"],
            augment["elastic"]["jitter_sigma"],
            [augment["elastic"]["rotation_min"]*np.pi/180.0,
             augment["elastic"]["rotation_max"]*np.pi/180.0],
            uniform_3d_rotation=is_3d,
            subsample=augment["elastic"].get("subsample", 1),
            spatial_dims=3 if is_3d else 2,
            temporal_dim=False)
         if augment.get("elastic") is not None else gp.NoOp()) +

        (gp.SimpleAugment(
            mirror_only=augment["simple"].get("mirror"),
            transpose_only=augment["simple"].get("transpose"))
         if augment.get("simple") is not None else gp.NoOp()) +

        nl.PermuteChannel(raw, config.get("probability_permute", 0)) +

        (nl.OverlayAugment(
            raw,
            gt_instances if overlapping_inst else gt_labels,
            apply_probability=augment["overlay"]["probability_overlay"],
            overlay_background=True,
            numinst=gt_numinst if overlapping_inst else gt_fgbg,
            max_numinst=config["max_num_inst"] if overlapping_inst else 1,
            loss_mask=loss_mask if add_partly else None
            ) if augment.get("overlay") is not None else gp.NoOp())  +

        (gp.ZeroPadChannels(
            gt_instances,
            gt_max_num_inst if augment.get("overlay") is None
            else gt_max_num_inst*2
        ) if overlapping_inst else gp.NoOp()) +

        nl.RandomHue(
            raw, config['hue_max_change'],
            config['probability_hue']
        ) +

        (gp.IntensityAugment(
            raw,
            scale_min=augment["intensity"]["scale"][0],
            scale_max=augment["intensity"]["scale"][1],
            shift_min=augment["intensity"]["shift"][0],
            shift_max=augment["intensity"]["shift"][1],
            z_section_wise=False,
            clip=False)
         if augment.get("intensity") is not None else gp.NoOp()) +

        (gp.IntensityScaleShift(raw, 2, -1)
            if config.get("shift_intensity", False) else gp.NoOp()) +

        (gp.Reject(
            gt_numinst if overlapping_inst else gt_fgbg,
            min_masked=config["reject_min_masked"],
            reject_probability=1,
            mask_filter=1 if overlapping_inst else None
            ) if config.get("reject_min_masked", 0) > 0 else gp.NoOp()) +

        # convert labels into affinities between voxels
        (gp.AddAffinities(
            neighborhood,
            gt_instances if overlapping_inst else gt_labels,
            gt_affs,
            multiple_labels=overlapping_inst,
            torch=config.get("add_affinities", "cpu") == "torch",
            dtype=np.float32)
         if add_affinities in ["cpu", "torch"] else gp.NoOp()) +

        (gp.PreCache(
            cache_size=config["cache_size"],
            num_workers=config["num_workers"])
         if config["num_workers"] > 1 else gp.NoOp())

        + gp.Stack(config["batch_size"])
    )

    # set up optional validation path without augmentations
    if config["val_log_step"] is not None:
        val_pipeline = (
            val_sources +

            (gp.IntensityScaleShift(raw, 2, -1)
                if config.get("shift_intensity", False) else gp.NoOp()) +

            (gp.Reject(
                gt_numinst if overlapping_inst else gt_fgbg,
                min_masked=config["reject_min_masked"],
                reject_probability=1,
                mask_filter=1 if overlapping_inst else None
                ) if config.get("reject_min_masked", 0) > 0 else gp.NoOp()) +

            (gp.ZeroPadChannels(
                gt_instances,
                gt_max_num_inst_val if augment.get("overlay") is None
                else gt_max_num_inst_val*2
            ) if overlapping_inst else gp.NoOp()) +

            # convert labels into affinities between voxels
            (gp.AddAffinities(
                neighborhood,
                gt_instances if overlapping_inst else gt_labels,
                gt_affs,
                multiple_labels=overlapping_inst,
                torch=config.get("add_affinities", "cpu") == "torch",
                dtype=np.float32)
             if add_affinities in ["cpu", "torch"] else gp.NoOp()) +

            (gp.PreCache(
                cache_size=config["batch_size"],
                num_workers=1)
             if config["num_workers"] > 1 else gp.NoOp())

            + gp.Stack(config["batch_size"])
        )

    if config["val_log_step"] is not None:
        pipeline = (
            (train_pipeline, val_pipeline) +
            gp.TrainValProvider(
                step=config["val_log_step"], init_step=trained_until)
        )
    else:
        pipeline = train_pipeline

    inputs = {
        'raw': raw,
    }
    if add_affinities in ["cpu", "torch"]:
        inputs['gt_affs'] = gt_affs
    else:
        inputs['gt_labels'] = gt_instances if overlapping_inst else gt_labels

    outputs = {
        0: pred_affs,
        # 2: raw_cropped,
        2: gt_affs_samples,
        3: gt_fgbg_loss
    }

    loss_inputs = {
        'pred_logits_affs': pred_affs,
        'gt_affs_samples': gt_affs_samples,
    }

    loss_outputs = {}

    gradients = {
        # 0: pred_affs_gradients,
    }

    snapshot_datasets = {
        raw: 'volumes/raw',
        # raw_cropped: 'volumes/raw_cropped',
        gt_affs_samples: '/volumes/gt_affs',
    }

    snapshot_datasets_dtypes = {}

    key_to_fun = {
        pred_affs: scipy.special.expit
    }

    if train_code:
        outputs[4] = pred_code
        snapshot_datasets[pred_code] = '/volumes/pred_code'
        # loss_outputs["pred_affs"] = pred_affs
        # loss_outputs["gt_affs_samples"] = gt_affs_samples
        # snapshot_datasets[gt_affs] = '/volumes/gt_affs_samples'
        # snapshot_datasets[pred_code_gradients] = '/volumes/pred_code_gradients'
        # gradients[4] = pred_code_gradients

    # else:
    snapshot_datasets[pred_affs] = '/volumes/pred_affs'
    # snapshot_datasets[pred_affs_gradients] = '/volumes/pred_affs_gradients'


    if overlapping_inst:
        inputs['gt_fgbg_numinst'] = gt_numinst
        loss_inputs['gt_fgbg_loss'] = gt_fgbg_loss
        outputs[1] = pred_numinst
        loss_inputs['pred_logits_fg'] = pred_numinst
        snapshot_datasets[pred_numinst] = '/volumes/pred_numinst'
        snapshot_datasets_dtypes[pred_numinst] = np.float32
        snapshot_datasets[gt_numinst] = '/volumes/gt_numinst'
        key_to_fun[pred_numinst] = lambda x: scipy.special.softmax(x, axis=1)
    else:
        inputs['gt_fgbg_numinst'] = gt_fgbg
        loss_inputs['gt_fgbg_loss'] = gt_fgbg_loss
        outputs[1] = pred_fgbg
        loss_inputs['pred_logits_fg'] = pred_fgbg
        snapshot_datasets[pred_fgbg] = '/volumes/pred_fgbg'
        snapshot_datasets_dtypes[pred_fgbg] = np.float32
        snapshot_datasets[gt_fgbg] = '/volumes/gt_fgbg'
        key_to_fun[pred_fgbg] = scipy.special.expit

    if add_partly:
        loss_inputs['loss_mask'] = loss_mask

    if logger.isEnabledFor(logging.INFO):
        logger.debug("requires_grad enabled for:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info("%s", name)
        # for name, param in decoder.named_parameters():
        #     if param.requires_grad:
        #         logger.debug("%s", name)

    # create loss object
    loss = torch_loss.LossWrapper(
        config, current_step=trained_until)

    # create optimizer
    opt_kwargs = {k: v for k,v in config["optimizer"].items()
                  if k != "optimizer" and k != "loss"}
    opt = getattr(torch.optim, config["optimizer"]["optimizer"])(
        model.parameters(), **opt_kwargs)

    hparam_dict = {
        'opt': config["optimizer"]["optimizer"],
        'wd': config["optimizer"].get("weight_decay"),
    }

    if config.get('lr_scheduler'):
        sched_kwargs = {k: v for k,v in config["lr_scheduler"].items()
                        if k != "lr_scheduler" and k != "lr_schedule_per_epoch"}
        lr_scheduler = getattr(
            torch.optim.lr_scheduler, config["lr_scheduler"]["lr_scheduler"])(
                opt, **sched_kwargs)
        lr_schedule_per_epoch = config["lr_scheduler"].get(
            "lr_schedule_per_epoch", False)
        hparam_dict["lr_sched"] = config['lr_scheduler']["lr_scheduler"]
        hparam_dict["max_lr"] = config['lr_scheduler']["max_lr"]
    else:
        lr_scheduler = None
        lr_schedule_per_epoch = None

    # if new training, save initial state to disk
    if trained_until == 0:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            },
            checkpoint_basename + "_checkpoint_0")

    # and add training gunpowder node
    pipeline = (
        pipeline +
        gp.TorchTrainExt(
            model=model,
            loss=loss,
            optimizer=opt,
            checkpoint_basename=checkpoint_basename,
            inputs=inputs,
            outputs=outputs,
            loss_inputs=loss_inputs,
            loss_outputs=loss_outputs,
            gradients=gradients,
            lr_scheduler=lr_scheduler,
            lr_schedule_per_epoch=lr_schedule_per_epoch,
            log_dir=os.path.join(config['output_folder'], ".."),
            val_log_step=config.get("val_log_step"),
            use_auto_mixed_precision=config.get("use_auto_mixed_precision", False),
            use_swa=config.get("use_swa"),
            swa_every_it=config.get("swa_every_it"),
            swa_start_it=config.get("swa_start_it"),
            swa_freq_it=config.get("swa_freq_it"),
            len_epoch=config.get("len_epoch"),
            save_every=config["checkpoints"],
            train_padding=config.get("train_padding"),
            val_padding=config.get("val_padding"),
            hparam_dict=hparam_dict,
            init_ssl=config.get("init_ssl", False),
            max_iterations=config["max_iterations"]) +

        gp.ApplyFun(key_to_fun,
                     every=config["snapshots"]) +
        # visualize
        gp.Snapshot(snapshot_datasets,
                     output_dir=os.path.join(config['output_folder'], 'snapshots'),
                     output_filename='snapshot_{iteration}.hdf',
                     additional_request=snapshot_request,
                     every=config["snapshots"],
                     # optional: cast some array to fiji readable type
                     dataset_dtypes=snapshot_datasets_dtypes,
                     init_step=trained_until
                     ) +
        gp.PrintProfilingStats(every=config["profiling"])
    )

    # finalize pipeline and start training
    with gp.build(pipeline):

        logger.info("Starting training...")
        with logging_redirect_tqdm():
            for i in tqdm(range(trained_until, config["max_iterations"]),
                          disable=None):
                start = time.time()
                if (val_request is not None and
                    config.get("val_log_step") and
                    i % config["val_log_step"] == 1):
                    # print("val", val_request)
                    with torch.no_grad():
                        pipeline.request_batch(val_request)
                else:
                    # print("train", request)
                    pipeline.request_batch(request)
                time_of_iteration = time.time() - start

                logger.info(
                    "Batch: iteration=%d, time=%f",
                    i+1, time_of_iteration)


def get_sources(config, arrays, voxel_size, data_files, val=False, graphs=None):
    """Create gunpowder source nodes for each data source in config

    Args
    ----
    config: TrackingConfig
        Configuration object
    arrays: List of gp.Array
        Data arrays will be stored here.
    data_files:
        List of files to load
    """

    overlapping_inst = config.get('overlapping_inst')
    sample_fg_points = config.get("sample_fg_points", False)

    raw = arrays['raw']
    gt_close_to_overlap = graphs['gt_close_to_overlap']
    if sample_fg_points:
        gt_fg_points = graphs["gt_fg_points"]
    else:
        gt_sample_mask = arrays['gt_sample_mask']

    overlap_csvs = {}
    overlap_folder = config.get("overlap_csv_folder")
    if sample_fg_points:
        fg_csvs = {}
        fg_csv_folder = config["fg_csv_folder"]

    for fn in data_files:
        basefn = os.path.basename(fn).replace(".zarr", ".csv")
        csv_fn = os.path.join(overlap_folder, basefn )
        if os.path.exists(csv_fn):
            overlap_csvs[fn] = csv_fn
        if sample_fg_points:
            fg_csvs[fn] = os.path.join(fg_csv_folder, basefn) 

    if config.get("add_partly"):
        data_files = data_files * config.get("oversample_complete", 1)
        dt = config["train_data"] if not val else config["val_data"]
        dt = dt.replace("complete", "partly")
        overlap_folder_partly = overlap_folder.replace("complete", "partly")
        if sample_fg_points:
            fg_csv_folder_partly = fg_csv_folder.replace("complete", "partly")
        # data_files = data_files[:1]
        for fn in os.listdir(dt):
            if ".zarr" not in fn:
                continue
            csv_fn = os.path.join(
                overlap_folder_partly, fn.replace(".zarr", ".csv"))
            fn = os.path.join(dt, fn)
            data_files.append(fn)
            if os.path.exists(csv_fn):
                overlap_csvs[fn] = csv_fn
            if sample_fg_points:
                fg_csvs[fn] = os.path.join(
                        fg_csv_folder_partly,
                        os.path.basename(fn).replace(".zarr", ".csv")
                        )

    sources = []
    sources_overlap = []
    sources_random = []

    if config['input_format'] == "hdf":
        sourceNode = gp.Hdf5Source
    elif config['input_format'] == "zarr":
        sourceNode = gp.ZarrSource
    else:
        raise NotImplementedError("train node for %s not implemented yet",
                                  config['input_format'])

    gt_max_num_inst = 0
    for idx, fn in enumerate(data_files):
        logger.info("loading data %s (val: %s)", fn, val)
        # if idx == 2:
        #     break
        if config['input_format'] == "hdf":
            with h5py.File(fn, 'r') as f:
                dtype = f[config["raw_key"]].dtype
                num_inst = f['volumes/gt_instances'].shape[0]
        else: #if config['input_format'] == "zarr":
            f = zarr.open(fn, 'r')
            dtype = f[config["raw_key"]].dtype
            num_inst = f['volumes/gt_instances'].shape[0]
            # get sample size for points roi
            sample_shape = f['volumes/gt_instances'].shape[1:]

        assert dtype == np.float32, "please convert data to float32!"
        # limit_to_roi = gp.Roi(offset=ds.roi.offset, shape=ds.roi.shape)
        # logger.info("limiting to roi: %s", limit_to_roi)
        if overlapping_inst:
            gt_max_num_inst = max(gt_max_num_inst, num_inst)

        datasets = {
            raw: config["raw_key"],
        }
        array_specs = {
            raw: gp.ArraySpec(
                interpolatable=True,
                voxel_size=voxel_size),
        }
        if overlapping_inst:
            gt_numinst = arrays['gt_numinst']
            datasets[gt_numinst] = 'volumes/gt_numinst'
            array_specs[gt_numinst] = gp.ArraySpec(interpolatable=False,
                    voxel_size=voxel_size)
            gt_instances = arrays['gt_instances']
            datasets[gt_instances] = config.get('gt_key', 'volumes/gt_instances')
            array_specs[gt_instances] = gp.ArraySpec(interpolatable=False,
                    voxel_size=voxel_size)
            if not sample_fg_points:
                datasets[gt_sample_mask] = 'volumes/gt_fg_rm_5'
                array_specs[gt_sample_mask] = gp.ArraySpec(interpolatable=False,
                        voxel_size=voxel_size)
        else:
            gt_fgbg = arrays['gt_fgbg']
            datasets[gt_fgbg] = 'volumes/gt_fg'
            array_specs[gt_fgbg] = gp.ArraySpec(interpolatable=False,
                    voxel_size=voxel_size)
            gt_labels = arrays['gt_labels']
            datasets[gt_labels] = config.get('gt_key', 'volumes/gt_labels')
            array_specs[gt_labels] = gp.ArraySpec(interpolatable=False,
                    voxel_size=voxel_size)

        if "partly" in fn:
            if config.get("use_gt_extra"):
                gt_fgbg_extra = arrays['gt_fgbg_extra']
                datasets[gt_fgbg_extra] = 'volumes/gt_fg_extra'
                array_specs[gt_fgbg_extra] = gp.ArraySpec(interpolatable=False,
                                                          voxel_size=voxel_size)
                if overlapping_inst:
                    mask_source = [gt_numinst, gt_fgbg_extra]
                else:
                    mask_source = [gt_fgbg, gt_fgbg_extra]
            elif overlapping_inst:
                mask_source = gt_numinst
            else:
                mask_source = gt_fgbg
        else:
            mask_source = None

        # get random locations source
        if config["sampling"].get("probability_random", 0) > 0 and \
           (not "partly" in fn or
            (config.get('mask_bg_weight', 0.0) != 0.0 and
             config.get("use_gt_extra"))
            ):
            print("random source: ", fn)
            file_source = sourceNode(
                    fn,
                    datasets=datasets,
                    array_specs=array_specs
                    )
            file_source = (
                    file_source +
                    gp.Pad(raw, None) +
                    gp.Pad(gt_instances if overlapping_inst else gt_labels, (70, 70, 70)) +
                    gp.Pad(gt_numinst if overlapping_inst else gt_fgbg, (70, 70, 70)) +
                    gp.RandomLocation() +
                    (gp.CreateMask(
                        arrays['loss_mask'],
                        gt_numinst if overlapping_inst else gt_fgbg,
                        mask_source=mask_source,
                        bg_weight=config.get('mask_bg_weight', 0.0),
                        use_gt_extra=config.get("use_gt_extra"))
                     if config.get("add_partly") else gp.NoOp())
                    )
            if overlapping_inst:
                file_source += gp.Cast(gt_numinst, dtype=np.int64)
            else:
                file_source += gp.Cast(gt_fgbg, dtype=np.float32)
            sources_random.append(file_source)

        # get foreground source
        file_source = sourceNode(
                fn,
                datasets=datasets,
                array_specs=array_specs
                )
        if sample_fg_points:
            file_source = (file_source,
                    gp.CsvPointsSource(
                        fg_csvs[fn], gt_fg_points,
                        gp.GraphSpec(roi=gp.Roi(offset=[0,0,0], shape=sample_shape)),
                        ndims=3)
                        )
            file_source = file_source + gp.MergeProvider()

        if overlapping_inst:
            if sample_fg_points:
                fg_random_location = gp.RandomLocation(
                        ensure_nonempty=gt_fg_points,
                        point_balance_radius=config["sampling"]["point_balance_radius_overlap"],
                        copy_roi_from=gt_instances
                        )
            else:
                fg_random_location = gp.RandomLocation(
                        min_masked=config["sampling"]["min_masked"],
                        mask=gt_sample_mask,
                        copy_roi_from=gt_instances
                        )
        else:
            fg_random_location = gp.RandomLocation()

        file_source = (
            file_source +
            gp.Pad(raw, None) +
            gp.Pad(gt_instances if overlapping_inst else gt_labels, (70, 70, 70)) +
            gp.Pad(gt_numinst if overlapping_inst else gt_fgbg, (70, 70, 70)) +
            # chose a random location for each requested batch
            fg_random_location +
            (gp.CreateMask(
                arrays['loss_mask'],
                gt_numinst if overlapping_inst else gt_fgbg,
                mask_source=mask_source,
                bg_weight=config.get('mask_bg_weight', 0.0),
                use_gt_extra=config.get("use_gt_extra"))
             if config.get("add_partly") else gp.NoOp())
        )
        if overlapping_inst:
            file_source += gp.Cast(gt_numinst, dtype=np.int64)
        else:
            file_source += gp.Cast(gt_fgbg, dtype=np.float32)
        sources.append(file_source)

        # get close to overlap source
        if overlapping_inst and fn in overlap_csvs:
            datasets_overlap = datasets.copy()
            array_specs_overlap = array_specs.copy()
            if not sample_fg_points:
                del datasets_overlap[gt_sample_mask]
                del array_specs_overlap[gt_sample_mask]
            file_source = (
                    sourceNode(
                        fn,
                        datasets=datasets_overlap,
                        array_specs=array_specs_overlap),
                    gp.CsvPointsSource(
                        overlap_csvs[fn], gt_close_to_overlap,
                        gp.GraphSpec(roi=gp.Roi(offset=[0,0,0], shape=sample_shape)),
                        ndims=3)
                        )
            file_source = file_source + gp.MergeProvider()

            file_source = (
                file_source +
                gp.Pad(raw, None) +
                gp.Pad(gt_instances if overlapping_inst else gt_labels, (70, 70, 70)) +
                gp.Pad(gt_numinst if overlapping_inst else gt_fgbg, (70, 70, 70)) +
                gp.Cast(gt_numinst, dtype=np.int64) +
                gp.RandomLocation(
                    ensure_nonempty=gt_close_to_overlap,
                    point_balance_radius=config["sampling"]["point_balance_radius_overlap"],
                    copy_roi_from=gt_instances
                    ) +
                 (gp.CreateMask(
                     arrays['loss_mask'],
                     gt_numinst if overlapping_inst else gt_fgbg,
                     mask_source=mask_source,
                     bg_weight=config.get('mask_bg_weight', 0.0),
                     use_gt_extra=config.get("use_gt_extra"))
                  if config.get("add_partly") else gp.NoOp())
            )
            sources_overlap.append(file_source)

    if overlapping_inst:
        source = (tuple(sources) + gp.RandomProvider(),
                tuple(sources_overlap) + gp.RandomProvider())
        probs = [config["sampling"]["probability_fg"],
                config["sampling"]["probability_overlap"]]

        if len(sources_random) > 0:
            source += (tuple(sources_random) + gp.RandomProvider(),)
            probs += [config["sampling"]["probability_random"]]

        # chose a random source (i.e., sample) from the above
        source = (
                source +
                gp.RandomProvider(probabilities=probs)
                )
    else:
        source = tuple(sources) + gp.RandomProvider()

    #if overlapping_inst:
    #    source += gp.ZeroPadChannels(gt_instances, gt_max_num_inst)

    return source, gt_max_num_inst
