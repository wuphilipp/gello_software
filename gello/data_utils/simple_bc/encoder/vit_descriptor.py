import math
import types
from typing import List, Tuple

import hydra
import torch
import torch.nn.modules.utils as nn_utils
from omegaconf import DictConfig
from torch import nn
from torchvision import transforms

from simple_bc._interfaces.encoder import Encoder
from simple_bc.utils.torch_utils import pack_one, unpack_one, to_torch, rearrange


class ViTDescriptor(Encoder):
    def __init__(self,
                 model_type: str = 'dino_vits8',
                 stride: int = 4,
                 layer: int = 9,
                 freeze_pretrained: bool = True,
                 downsample: bool = False,
                 use_cached_token: bool = False,
                 patch_size: int = 0,  # Not used, but keep it here so patch size can be passed in
                 use_pretrained=True,
                 **kwargs
                 ):
        """
        :param model_type: A string specifying the type of model to extract from
        :param stride: stride of first convolution layer. small stride -> higher resolution
        :param layer: the layer to extract from. 0 is the first layer after the input
        """
        super(ViTDescriptor, self).__init__(**kwargs)
        self.model_type = model_type
        self.use_pretrained=use_pretrained
        if not use_pretrained:
            assert not freeze_pretrained
        self.vit = self.build_vit(model_type, freeze_pretrained, use_pretrained)

        # TODO Note this is not True for MVP
        self.mean = (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        self.std = (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        if freeze_pretrained:
            for param in self.vit.parameters():
                param.requires_grad = False
        else:
            for param in self.vit.parameters():
                param.requires_grad = True
        self.downsample = downsample
        if 'clip' not in model_type:
            self.vit = ViTDescriptor.patch_vit_resolution(self.vit, stride=stride)
            self.p = self.vit.patch_embed.patch_size
            self.stride = self.vit.patch_embed.proj.stride
            if not isinstance(self.p, int):
                self.p = self.p[0]
        else:
            self.p = self.vit.p
            self.stride = self.vit.stride

        self.layer = layer

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def build_vit(model_type: str, freeze_pretrained, use_pretrained) -> nn.Module:
        if 'dinov2' in model_type:
            model = torch.hub.load('facebookresearch/dinov2', model_type, pretrained=use_pretrained)
        elif 'dino' in model_type:
            model = torch.hub.load('facebookresearch/dino:main', model_type, pretrained=use_pretrained)
        elif 'mvp' in model_type:
            assert use_pretrained
            model_type = model_type.replace('mvp_', '')
            import mvp
            model = mvp.load(model_type)
            if freeze_pretrained:
                model.freeze()
        return model

    @staticmethod
    def prune_model(model: nn.Module, layer: int = 9) -> nn.Module:
        """
        :param model: the model to prune
        :param layer: the layer to extract from. 0 is the first layer after the input
        :return: the pruned model
        """
        try:
            model.transformer.resblocks = model.transformer.resblocks[:layer + 1]
        except AttributeError:
            model.blocks = model.blocks[:layer + 1]
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """

        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2, f'patch_size should be a tuple of length 2, got {patch_size}'
            assert patch_size[0] == patch_size[1]
            patch_size = patch_size[0]

        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)

        assert all(
            [(patch_size // s_) * s_ == patch_size for s_ in
             stride]), f'stride {stride} should divide patch_size {patch_size}'

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(ViTDescriptor._fix_pos_enc(patch_size, stride), model)
        return model

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)

            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx])  # Bxhxtxd

        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.vit.blocks):
            if block_idx in layers:
                if facet == 'token':
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == 'attn':
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                elif facet in ['key', 'query', 'value']:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch: torch.Tensor, layers: List[int] = 11, facet: str = 'key') -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.vit(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats

    def _extract_clip_feature(self, batch: torch.Tensor):
        B, C, H, W = batch.shape
        self.num_patches = (H // self.p[0], W // self.p[1])
        return self.vit.forward(batch)

    def extract_descriptors(self, batch: torch.Tensor, layer=None) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        if 'clip' in self.model_type:
            facet = 'token'
            x = self._extract_clip_feature(batch)
        else:
            facet = 'key' if ('mvp' not in self.model_type) else 'token'
            if layer is None:
                self._extract_features(batch, [self.layer], facet)
            else:
                self._extract_features(batch, [layer], facet)
            x = self._feats[0]

        if facet == 'token':
            x = x.unsqueeze(dim=1)  # Bx1xtxd

        desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)
        desc = desc / desc.norm(dim=-1, keepdim=True)

        cls, patch_embed = desc[:, :, 0], desc[:, :, 1:]
        return cls, patch_embed

    def preprocess(self, obs):
        with torch.no_grad():
            ret = {}
            if 'tokens' in obs:
                ret['tokens'], ret['cls'], ret['proprio'] = obs['tokens'], obs['cls'], obs['state']
            else:
                if "rgb" in obs and 'rgb' in self.obs_shapes:
                    # normalize RGB image
                    rgb, ps = pack_one(obs["rgb"], '* c h w')  # [0, 255]
                    rgb = rgb.float() / 255.0
                    rgb = self.normalize(rgb) # use imagenet mean and std
                    rgb = unpack_one(rgb, ps, '* c h w')
                    # debug
                    ret['video'] = rgb

            if "depth" in obs and 'depth' in self.obs_shapes:
                depth, ps = pack_one(obs["depth"], '* c h w')
              #  depth = torch.clip(depth, 0.0, 2.0); should already be normalized
                depth = unpack_one(depth, ps, '* c h w')
                ret['depth'] = depth.float()

            if 'state' in obs and 'state' in self.obs_shapes:
                ret['proprio'] = obs['state']
        return ret

    def forward_vit(self, video, **kwargs):
        """ Video shape: (B, F, C, H, W)
            Token shape: (B, F, D, pH, pW)
        """
        images, packed_shape = pack_one(video, '* c h w')
        cls, patch_embed = self.extract_descriptors(images)  # (*) x 1 x (pH x pW) x D
        cls = cls.squeeze(dim=1)
        patch_embed = patch_embed.squeeze(dim=1)
        P = int(math.sqrt(patch_embed.shape[-2]))
        tokens = rearrange(patch_embed, 'b (pH pW) d -> b d pH pW', pH=P, pW=P)
        if self.downsample:
            tokens = nn.AvgPool2d(5, 5)(tokens)
        tokens = unpack_one(tokens, packed_shape, '* c h w')
        cls = unpack_one(cls, packed_shape, '* c')
        return tokens, cls

    def forward(self, obs):
        # V: Number of views
        # tokens: B x F x V x D x H x W
        processed_obs = self.preprocess(obs)
        if 'tokens' in obs:
            tokens, cls = processed_obs['tokens'], processed_obs['cls']
        else:
            tokens, cls = self.forward_vit(processed_obs['video'])
        processed_obs['cls'] = cls
        return tokens, processed_obs