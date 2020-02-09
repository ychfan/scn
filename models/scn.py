from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import functools

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

import common.metrics
import models


def update_argparser(parser):
  models.update_argparser(parser)
  args, _ = parser.parse_known_args()
  parser.add_argument(
      '--num_blocks',
      help='Number of residual blocks in networks.',
      default=16,
      type=int)
  parser.add_argument(
      '--num_residual_units',
      help='Number of residual units in networks.',
      default=32,
      type=int)
  parser.add_argument(
      '--width_multiplier',
      help='Width multiplier inside residual blocks.',
      default=4,
      type=float)
  parser.add_argument(
      '--temporal_size',
      help='Number of frames for burst input.',
      default=None,
      type=int)
  parser.add_argument(
      '--num_scales',
      help='Number of scales in networks.',
      required=True,
      type=int)
  if args.dataset.startswith('div2k'):
    parser.set_defaults(
        train_epochs=30,
        learning_rate_milestones=(20, 25),
        learning_rate_decay=0.2,
        save_checkpoints_epochs=1,
        lr_patch_size=48,
        train_temporal_size=1,
        eval_temporal_size=1,
    )
  else:
    raise NotImplementedError('Needs to tune hyper parameters for new dataset.')


def get_model_spec(params):
  model = MODEL(params)
  print('# of parameters: ', sum([p.numel() for p in model.parameters()]))
  optimizer = optim.Adam(model.parameters(), params.learning_rate)
  lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                params.learning_rate_milestones,
                                                params.learning_rate_decay)
  loss_fn = torch.nn.L1Loss()
  metrics = {
      'loss': loss_fn,
      'PSNR': functools.partial(common.metrics.psnr, shave=params.scale + 6),
      'PSNR_Y': functools.partial(common.metrics.psnr_y, shave=params.scale),
  }
  return model, loss_fn, optimizer, lr_scheduler, metrics


class MODEL(nn.Module):

  def __init__(self, params):
    super(MODEL, self).__init__()
    self.temporal_size = params.temporal_size
    self.image_mean = params.image_mean
    kernel_size = 3
    skip_kernel_size = 5
    weight_norm = torch.nn.utils.weight_norm
    num_inputs = params.num_channels
    if self.temporal_size:
      num_inputs *= self.temporal_size
    num_outputs = params.scale * params.scale * params.num_channels
    self.num_scales = params.num_scales

    body = []
    conv = weight_norm(
        nn.Conv2d(
            num_inputs,
            params.num_residual_units,
            kernel_size,
            padding=kernel_size // 2))
    init.ones_(conv.weight_g)
    init.zeros_(conv.bias)
    body.append(conv)
    body.append(Head(params.num_residual_units, params.num_scales))
    for _ in range(params.num_blocks):
      body.append(
          Block(
              params.num_residual_units,
              kernel_size,
              params.width_multiplier,
              weight_norm=weight_norm,
              res_scale=1 / math.sqrt(params.num_blocks),
          ))
    body.append(Tail(params.num_residual_units))
    conv = weight_norm(
        nn.Conv2d(
            params.num_residual_units,
            num_outputs,
            kernel_size,
            padding=kernel_size // 2))
    init.ones_(conv.weight_g)
    init.zeros_(conv.bias)
    body.append(conv)
    self.body = nn.Sequential(*body)

    skip = []
    if num_inputs != num_outputs:
      conv = weight_norm(
          nn.Conv2d(
              num_inputs,
              num_outputs,
              skip_kernel_size,
              padding=skip_kernel_size // 2))
      init.ones_(conv.weight_g)
      init.zeros_(conv.bias)
      skip.append(conv)
    self.skip = nn.Sequential(*skip)

    shuf = []
    if params.scale > 1:
      shuf.append(nn.PixelShuffle(params.scale))
    self.shuf = nn.Sequential(*shuf)

  def forward(self, x):
    if self.temporal_size:
      x = x.view([x.shape[0], -1, x.shape[3], x.shape[4]])
    x -= self.image_mean
    skip = self.skip(x)
    x_shape = x.shape
    is_padding = False
    if x.shape[-1] % (2**self.num_scales) or x.shape[-2] % (2**self.num_scales):
      pad_h = (-x.shape[-2]) % (2**self.num_scales)
      pad_w = (-x.shape[-1]) % (2**self.num_scales)
      x = nn.functional.pad(x, (0, pad_w, 0, pad_h), 'replicate')
      is_padding = True
    x = self.body(x)
    if is_padding:
      x = x[..., :x_shape[-2], :x_shape[-1]]
    x = x + skip
    x = self.shuf(x)
    x += self.image_mean
    if self.temporal_size:
      x = x.view([x.shape[0], -1, 1, x.shape[2], x.shape[3]])
    return x


class Head(nn.Module):

  def __init__(self, n_feats, num_scales):
    super(Head, self).__init__()
    self.num_scales = num_scales

    down = []
    down.append(nn.UpsamplingBilinear2d(scale_factor=0.5))
    self.down = nn.Sequential(*down)

  def forward(self, x):
    x_list = [x]
    for _ in range(self.num_scales - 1):
      x_list.append(self.down(x_list[-1]))
    return x_list


class Block(nn.Module):

  def __init__(self,
               num_residual_units,
               kernel_size,
               width_multiplier=1,
               weight_norm=torch.nn.utils.weight_norm,
               res_scale=1):
    super(Block, self).__init__()
    body = []
    conv = weight_norm(
        nn.Conv2d(
            num_residual_units,
            int(num_residual_units * width_multiplier),
            kernel_size,
            padding=kernel_size // 2))
    init.constant_(conv.weight_g, 2.0)
    init.zeros_(conv.bias)
    body.append(conv)
    body.append(nn.ReLU(True))
    conv = weight_norm(
        nn.Conv2d(
            int(num_residual_units * width_multiplier),
            num_residual_units,
            kernel_size,
            padding=kernel_size // 2))
    init.constant_(conv.weight_g, res_scale)
    init.zeros_(conv.bias)
    body.append(conv)

    self.body = nn.Sequential(*body)

    down = []
    down.append(
        weight_norm(nn.Conv2d(num_residual_units, num_residual_units, 1)))
    down.append(nn.UpsamplingBilinear2d(scale_factor=0.5))
    self.down = nn.Sequential(*down)

    up = []
    up.append(weight_norm(nn.Conv2d(num_residual_units, num_residual_units, 1)))
    up.append(nn.UpsamplingBilinear2d(scale_factor=2.0))
    self.up = nn.Sequential(*up)

  def forward(self, x_list):
    res_list = [self.body(x) for x in x_list]
    down_res_list = [res_list[0]] + [self.down(x) for x in res_list[:-1]]
    up_res_list = [self.up(x) for x in res_list[1:]] + [res_list[-1]]
    x_list = [
        x + r + d + u
        for x, r, d, u in zip(x_list, res_list, down_res_list, up_res_list)
    ]
    return x_list


class Tail(nn.Module):

  def __init__(self, n_feats):
    super(Tail, self).__init__()

  def forward(self, x_list):
    return x_list[0]