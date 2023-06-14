"""Provides a U-Net based tracking model class using torch
"""
import json
import logging
import os

import monai
import numpy as np
import torch
import torchinfo

from funlib.learn.torch.models import UNet, ConvPass, Downsample, Upsample

from PatchPerPix.util import (
    crop,
    crop_to_factor,
    gather_nd_torch,
    gather_nd_torch_no_batch,
    # seg_to_affgraph_3d_multi_torch_code_batch,
    # seg_to_affgraph_2d_multi_torch_code_batch,
    seg_to_affgraph_3d_multi_torch_code,
    seg_to_affgraph_2d_multi_torch_code,
    seg_to_affgraph_3d_multi_torch,
    seg_to_affgraph_2d_multi_torch,
    seg_to_affgraph_3d_torch,
    seg_to_affgraph_2d_torch,
    seg_to_affgraph_3d_torch_code,
    seg_to_affgraph_2d_torch_code,
)

logger = logging.getLogger(__name__)


class UnetModelWrapper(torch.nn.Module):
    """Wraps a torch U-Net implementation and extends it

    Supports multiple styles of U-Nets:
    - a single network for both code/affs and fgbg/numinst (single)
    - two separate networks (split)
    - a shared encoder and two decoders (multihead)

    Adds a layer to directly perform non maximum suppression using a 3d
    pooling layer with stride 1

    Input and Output shapes can be precomputed using `inout_shapes` (they
    can differ as valid padding is used by default)
    """
    def __init__(self, config, device, current_step=0, for_inference=False):
        super().__init__()
        self.config = config
        self.device = device
        self.current_step = current_step
        self.train_code = config.get("train_code")
        self.overlapping_inst = config.get('overlapping_inst')
        self.patchshape_squeezed = tuple(p for p in self.config['patchshape']
                                         if p > 1)
        self.patchsize = int(np.prod(self.patchshape_squeezed))
        self.ps = self.patchshape_squeezed[0]
        self.psH = self.ps // 2

        num_fmaps = config["num_fmaps"]
        self.num_channels = config.get("num_channels", 1)

        self.spatial_dims = len(config.get("train_input_shape",
                                           config["train_input_shape_valid"]))
        if isinstance(config['kernel_size'], int):
            ks = [config['kernel_size']] * self.spatial_dims
            ks = [ks] * config['num_repetitions']
            ks = [ks] * (len(config["downsample_factors"])+1)
        else:
            ks = config['kernel_size']

        if config['patch_activation'] == 'relu':
            config['patch_activation'] = "ReLU"
        if config['patch_activation'] == 'sigmoid':
            config['patch_activation'] = "Sigmoid"

        if for_inference:
            input_shape = (config["test_input_shape_valid"]
                           if config["val_padding"] == "valid"
                           else config["test_input_shape_same"])
            padding = config["val_padding"]
        else:
            input_shape = (config["train_input_shape_valid"]
                           if config["train_padding"] == "valid"
                           else config["train_input_shape_same"])
            padding = config["train_padding"]
        if self.config.get("network_style", "unet").lower() == "unet":
            self.unet = UNet(
                in_channels=self.num_channels,
                num_fmaps=num_fmaps,
                fmap_inc_factor=config["fmap_inc_factors"],
                fmap_dec_factor=config["fmap_dec_factors"],
                downsample_factors=config["downsample_factors"],
                kernel_size_down=ks,
                kernel_size_up=ks,
                upsampling=config["upsampling"],
                padding=padding,
                num_heads=1,
            )
            print(self.unet)
            torchinfo.summary(
                self.unet,
                input_size=((self.config.get("batch_size", 1), self.num_channels) +
                            tuple(input_shape)),
                depth=16,
                mode="train",
                verbose=1)

        elif self.config.get("network_style", "unet").lower() == "swinunetr":
            self.unet = monai.networks.nets.SwinUNETR(
                input_shape,
                in_channels=self.num_channels,
                out_channels=3,
                feature_size=num_fmaps,
                fmap_dec_factor=config["fmap_dec_factors"][-1]
            )
        else:
            raise RuntimeError(
                "invalid network style: %s",
                config["network_style"])

        if self.train_code:
            out_fm_code_affs = config["code_units"]
        else:
            out_fm_code_affs = self.patchsize
        if self.overlapping_inst:
            out_fm_fgbg_numinst = config['max_num_inst'] + 1
        else:
            out_fm_fgbg_numinst = 1
        # no padding for 1x1 conv
        self.out_code_affs = ConvPass(
            self.unet.out_channels, out_fm_code_affs,
            [[1] * self.spatial_dims], activation=None,
            padding=0)
        self.out_fgbg_numinst = ConvPass(
            self.unet.out_channels, out_fm_fgbg_numinst,
            [[1] * self.spatial_dims], activation=None,
            padding=0)

        if self.train_code:
            ae_config = config['autoencoder']
            ae_config['input_shape_squeezed'] = self.patchshape_squeezed
            ae_config['code_units'] = config['code_units']
            self.decoder = Autoencoder(ae_config)
            self.code_activation = getattr(
                torch.nn,
                config['autoencoder'].get('code_activation', 'Identity'))()
            self.sample_cnt = ae_config.get("num_code_samples", 1024)

            torchinfo.summary(
                self.decoder,
                input_size=((self.config.get("batch_size", 1),
                             ae_config['code_fmaps']) +
                            (self.decoder.code_spatial,)* self.spatial_dims),
                depth=16,
                mode="train",
                verbose=1)
        else:
            self.patch_activation = getattr(
                torch.nn,
                config.get('patch_activation', 'Sigmoid'))()


        neighborhood = []
        if self.train_code:
            for k in range(config['patchshape'][0]):
                for i in range(config['patchshape'][1]):
                    for j in range(config['patchshape'][2]):
                        if len(config["voxel_size"]) == 3:
                            neighborhood.append([k, i, j])
                        else:
                            neighborhood.append([i,j])
        else:
            psH = np.array(config['patchshape'])//2
            for k in range(-psH[0], psH[0]+1, config['patchstride'][0]):
                for i in range(-psH[1], psH[1]+1, config['patchstride'][1]):
                    for j in range(-psH[2], psH[2]+1, config['patchstride'][2]):
                        if len(config["voxel_size"]) == 3:
                            neighborhood.append([k, i, j])
                        else:
                            neighborhood.append([i,j])
            self.padding_neg = list(
                min([0] + [a[d] for a in neighborhood])
                for d in range(len(config["voxel_size"])))
            self.padding_pos = list(
                max([0] + [a[d] for a in neighborhood])
                for d in range(len(config["voxel_size"])))
            print(self.padding_neg, self.padding_pos)
        self.neighborhood = torch.as_tensor(neighborhood, device=self.device)
        print(self.neighborhood)

        if self.train_code:
            self.b_columns = []
            for b in range(self.config["batch_size"]):
                self.b_columns.append(
                    torch.full(
                        (self.sample_cnt, 1), b, dtype=torch.int32,
                        device=self.device))

        if self.config.get("add_affinities", "cpu") == "loss":
            if self.overlapping_inst:
                if len(config["voxel_size"]) == 3:
                    if self.train_code:
                        self.compute_affinities = seg_to_affgraph_3d_multi_torch_code
                    else:
                        self.compute_affinities = seg_to_affgraph_3d_multi_torch
                else:
                    if self.train_code:
                        self.compute_affinities = seg_to_affgraph_2d_multi_torch_code
                    else:
                        self.compute_affinities = seg_to_affgraph_2d_multi_torch
            else:
                if len(config["voxel_size"]) == 3:
                    if self.train_code:
                        self.compute_affinities = seg_to_affgraph_3d_torch_code
                    else:
                        self.compute_affinities = seg_to_affgraph_3d_torch
                else:
                    if self.train_code:
                        self.compute_affinities = seg_to_affgraph_2d_torch_code
                    else:
                        self.compute_affinities = seg_to_affgraph_2d_torch

    def init_layers(self):
        # the default init in pytorch is a bit strange
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
        # some modified (for backwards comp) version of kaiming
        # breaks training, cell_ind -> 0
        # activation func relu
        def init_weights(m):
            if isinstance(m, (
                    torch.nn.Conv3d, torch.nn.Conv2d,
                    torch.nn.ConvTranspose3d, torch.nn.ConvTranspose2d)):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        # activation func sigmoid
        def init_weights_sig(m):
            if isinstance(m, (
                    torch.nn.Conv3d, torch.nn.Conv2d,
                    torch.nn.ConvTranspose3d, torch.nn.ConvTranspose2d)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        logger.info("initializing model..")
        self.apply(init_weights)
        self.out_code_affs.apply(init_weights_sig)
        self.out_fgbg_numinst.apply(init_weights)

    def set_padding(self, padding):
        def set_pad_for_module(m):
            if isinstance(m, (torch.nn.Conv3d, torch.nn.Conv2d, Upsample)):
                m.padding = padding

        logger.info("setting padding to %s", padding)
        self.unet.apply(set_pad_for_module)
        self.out_code_affs.apply(set_pad_for_module)
        self.out_fgbg_numinst.apply(set_pad_for_module)

    def inout_shapes(self, input_shape, name, training):
        logger.info("getting output shape by running model")
        flipped = False
        if not self.training and training:
            flipped = True
            self.train()
        elif self.training and not training:
            flipped = True
            self.eval()
        with torch.no_grad():
            trial_run = self.out_code_affs(self.unet(
                torch.zeros([1, self.num_channels] + input_shape,
                            dtype=torch.float32).to(self.device)))
        if flipped:
            if self.training:
                self.eval()
            else:
                self.train()
        logger.info("done")
        output_shape = trial_run.size()[-self.spatial_dims:]
        net_config = {
            'input_shape': input_shape,
            'output_shape': output_shape
        }
        with open(
                os.path.join(self.config['output_folder'],
                             name + '_config.json'), 'w') as f:
            json.dump(net_config, f)

        logger.info(
            "input shape %s, output shape %s", input_shape, output_shape)

        return input_shape, output_shape

    def forward(self, raw, gt_affs=None, gt_fgbg_numinst=None, gt_labels=None):

        if len(raw.size()) == len(self.patchshape_squeezed) + 1:
            raw = torch.unsqueeze(raw, dim=1)
        model_out = self.unet(raw)

        pred_code_affs_logits = self.out_code_affs(model_out)
        pred_fgbg_numinst_logits = self.out_fgbg_numinst(model_out)

        if gt_fgbg_numinst is None:
            if self.train_code:
                pred_code_affs_logits = self.code_activation(pred_code_affs_logits)
            else:
                pred_code_affs_logits = self.patch_activation(pred_code_affs_logits)
            if self.overlapping_inst:
                pred_fgbg_numinst_logits = torch.softmax(
                    pred_fgbg_numinst_logits, dim=1)
            else:
                pred_fgbg_numinst_logits = torch.sigmoid(
                    pred_fgbg_numinst_logits)
            return pred_code_affs_logits, pred_fgbg_numinst_logits

        # nD b c? (d) w h
        if self.overlapping_inst:
            gt_fgbg = torch.clamp(gt_fgbg_numinst, 0, self.config['max_num_inst'])
            gt_fg = gt_fgbg == 1
            #gt_fgbg = gt_fgbg[
            #gt_fgbg = torch.nn.functional.one_hot(gt_fgbg, self.config['max_num_inst'] + 1)
            #gt_fgbg = torch.moveaxis(gt_fgbg, -1, 1).float()
            #gt_fgbg = torch.squeeze(gt_fgbg, dim=1)
        else:
            gt_fgbg = gt_fgbg_numinst
            gt_fg = gt_fgbg
        gt_fg = torch.squeeze(gt_fg, dim=1)

        # raw_cropped = crop(raw, code_affs.size()[-self.spatial_dims:]
        if self.train_code:
            pred_code = self.code_activation(pred_code_affs_logits)
            # nD b c (d) w h
            code = torch.movedim(pred_code, 1, -1)
            sample_cnt = self.sample_cnt

            if self.config.get("add_affinities", "cpu") == "loss":
                samples_locs = []
                for b in range(self.config["batch_size"]):
                    # list of coordinates
                    gt_fg_loc = torch.nonzero(gt_fg[b], as_tuple=False)
                    if gt_fg_loc.size(dim=0) == 0:
                        continue
                    # list if random ints
                    rand_loc_ids = torch.randint(
                        low=0,
                        high=gt_fg_loc.size(dim=0),
                        size=(min(sample_cnt, gt_fg_loc.size(dim=0)),),
                        dtype=torch.long,
                        device=self.device)

                    # use rand ints as index in coordinates list
                    samples_loc = gt_fg_loc[rand_loc_ids]
                    samples_loc = torch.cat(
                        [self.b_columns[b][:samples_loc.size()[0]], samples_loc],
                        dim=1)
                    samples_locs.append(samples_loc)
                    # get codes at selected coordinates

                if len(samples_locs) > 0:
                    samples_locs = torch.cat(samples_locs)
                    code_samples = gather_nd_torch_no_batch(code, samples_locs)
                    gt_affs = self.compute_affinities(
                        gt_labels, self.neighborhood, self.device, 
                        self.ps, self.psH,
                        samples_locs
                        )

                    gt_affs = torch.reshape(
                        gt_affs, [-1, 1] + list(self.patchshape_squeezed))
                    pred_affs_logits = self.decoder(code_samples)
                else:
                    pred_affs_logits = torch.zeros(
                        (self.config["batch_size"], 0,) +
                        pred_code_affs_logits.size()[2:],
                        device=self.device)
                    gt_affs = torch.zeros(
                        (self.config["batch_size"], 0,) +
                        pred_code_affs_logits.size()[2:],
                        device=self.device)

            else:
                code_samples = []
                gt_affs_samples = []
                for b in range(self.config["batch_size"]):
                    # list of coordinates
                    gt_fg_loc = torch.nonzero(gt_fg[b], as_tuple=False)
                    if gt_fg_loc.size(dim=0) == 0:
                        continue
                    # list if random ints
                    rand_loc_ids = torch.randint(
                        low=0,
                        high=gt_fg_loc.size(dim=0),
                        size=(min(sample_cnt, gt_fg_loc.size(dim=0)),),
                        dtype=torch.long,
                        device=self.device)

                    # use rand ints as index in coordinates list
                    samples_loc = gt_fg_loc[rand_loc_ids]
                    # get codes at selected coordinates
                    code_sample = gather_nd_torch_no_batch(code[b], samples_loc)
                    gt_affs_sample = gather_nd_torch_no_batch(
                        torch.movedim(gt_affs[b], 0, -1), samples_loc)

                    code_samples.append(code_sample)
                    gt_affs_samples.append(gt_affs_sample)

                if len(code_samples) > 0:
                    code_samples = torch.cat(code_samples)
                    gt_affs = torch.cat(gt_affs_samples)
                    gt_affs = torch.reshape(
                        gt_affs, [-1, 1] + list(self.patchshape_squeezed))
                    pred_affs_logits = self.decoder(code_samples)
                else:
                    pred_affs_logits = torch.zeros(
                        (self.config["batch_size"], 0,) +
                        pred_code_affs_logits.size()[2:],
                        device=self.device)
                    gt_affs = torch.zeros(
                        (self.config["batch_size"], 0,) +
                        pred_code_affs_logits.size()[2:],
                        device=self.device)
        else:
            pred_affs_logits = pred_code_affs_logits
            if self.config.get("add_affinities", "cpu") == "loss":
                gt_affs = self.compute_affinities(
                    gt_labels, self.neighborhood, self.device)
                if len(self.config["voxel_size"]) == 3:
                    gt_affs = gt_affs[
                        :, :,
                        self.padding_pos[0]:self.padding_neg[0],
                        self.padding_pos[1]:self.padding_neg[1],
                        self.padding_pos[2]:self.padding_neg[2]]
                else:
                    gt_affs = gt_affs[
                        :, :,
                        self.padding_pos[0]:self.padding_neg[0],
                        self.padding_pos[1]:self.padding_neg[1]]

        outputs = [
            pred_affs_logits, pred_fgbg_numinst_logits, gt_affs, gt_fgbg]

        if self.train_code:
            outputs.append(pred_code)

        return tuple(outputs)


class Autoencoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config['activation'] == 'relu':
            config['activation'] = "ReLU"
        if config['code_activation'] == 'relu':
            config['code_activation'] = "ReLU"
        if config['code_activation'] == 'sigmoid':
            config['code_activation'] = "Sigmoid"

        num_channels = config.get('num_channels', 1)

        self.down_conv = []
        self.down = []
        self.spatial_dims = len(config["input_shape_squeezed"])
        # spatial_dims = sum([1 for p in config['patchshape'] if p != 1])
        ks = [config['kernel_size']] * self.spatial_dims
        ks = [ks] * config['num_repetitions']
        nf_prev = num_channels
        for idx, nf in enumerate(config['num_fmaps']):
            self.down_conv.append(
                ConvPass(nf_prev, nf, ks,
                         activation=config['activation'],
                         padding=config['padding']))
            self.down.append(Downsample(config['downsample_factors'][idx]))
            nf_prev = nf
        self.down = torch.nn.ModuleList(self.down)
        self.down_conv = torch.nn.ModuleList(self.down_conv)

        self.code_fmaps = config['code_fmaps']
        self.code_units = config["code_units"]
        if self.spatial_dims == 2:
            self.code_spatial = int(np.sqrt(self.code_units/self.code_fmaps))
        else:
            self.code_spatial = int(np.cbrt(self.code_units/self.code_fmaps))
        assert np.power(self.code_spatial, self.spatial_dims) * self.code_fmaps == self.code_units, (
            "size of spatially reshaped code has to add up to code units")
        self.code_shape = (-1, self.code_fmaps) + (self.code_spatial,) * self.spatial_dims
        self.to_code = ConvPass(
            nf_prev, self.code_fmaps, [[1] * self.spatial_dims],
            activation=config['code_activation'],
            padding=config['padding'])

        self.from_code = ConvPass(
            self.code_fmaps, nf_prev, [[1] * self.spatial_dims],
            activation=config['activation'],
            padding=config['padding'])

        self.up = []
        self.up_conv = []
        for idx, nf in enumerate(list(reversed(config['num_fmaps']))[1:] + [1]):
            self.up.append(
                Upsample(config['downsample_factors'][-idx],
                         mode=config['upsampling'],
                         in_channels=nf_prev,
                         out_channels=nf,
                         activation=config['activation'],
                         padding=config['padding'],
                ))
            self.up_conv.append(
                ConvPass(nf, nf, ks,
                         activation=None if nf == 1 else config['activation'],
                         padding=config['padding']))
            nf_prev = nf
        self.up = torch.nn.ModuleList(self.up)
        self.up_conv = torch.nn.ModuleList(self.up_conv)
        print(self)

    def forward(self, code):
        # print(code.size())
        out = code

        # # down
        # for i in range(len(self.down)):
        #     out = self.down[i](out)
        #     out = self.down_conv[i](out)
        # print(out.size())
        # out = self.to_code(out)

        # print(out.size())
        # out = torch.reshape(code, (-1, 1) + self.config['input_shape_squeezed'])
        out = torch.reshape(out, self.code_shape)
        # print(code.size())
        out = self.from_code(out)
        for i in range(len(self.up)):
            out = self.up[i](out)
            out = self.up_conv[i](out)

        out = crop(out, self.config["input_shape_squeezed"])
        return out
