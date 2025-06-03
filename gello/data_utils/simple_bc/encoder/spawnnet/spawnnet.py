import math
from typing import List, Tuple
from simple_bc.encoder.r3m_encoder import load_r3m

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from einops import rearrange
from simple_bc._interfaces.encoder import Encoder
from simple_bc.utils.torch_utils import get_named_trainable_params, pack_one, unpack_one

"""
ViT feature extraction largely based on 'Deep ViT Features as Dense Visual Descriptors':
https://github.com/ShirAmir/dino-vit-features/blob/main/extractor.py
"""


class SpawnNet(Encoder):
    def __init__(
        self,
        conv_cfg: DictConfig,  # 'token_5':1, 'token_8':2, 'token_11':3
        pretrained_feat_info: dict,
        model_type="dino_vits8",
        stride=8,
        freeze_pretrained=False,
        freeze_vit_to_random=False,
        **kwargs,
    ):
        """
        :param model_type: A string specifying the type of model to extract from
        """
        super(SpawnNet, self).__init__(**kwargs)
        self.model_type = model_type
        self.conv_cfg = conv_cfg
        self.vit = self.build_vit(model_type)
        self.conv = self.build_conv(conv_cfg, pretrained_feat_info)
        self.out_shape = self.conv.out_shape
        self.mean = (
            (0.485, 0.456, 0.406)
            if ("dino" in self.model_type or "r3m" in self.model_type)
            else (0.5, 0.5, 0.5)
        )
        self.std = (
            (0.229, 0.224, 0.225)
            if ("dino" in self.model_type or "r3m" in self.model_type)
            else (0.5, 0.5, 0.5)
        )
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        print("Freezing ViT to random:", freeze_vit_to_random)

        self.freeze_vit_to_random = freeze_vit_to_random

        if freeze_pretrained:
            for param in self.vit.parameters():
                param.requires_grad = False
        else:
            for param in self.vit.parameters():
                param.requires_grad = True

        if "r3m" not in model_type:
            self.p = self.vit.patch_embed.patch_size
            self.stride = [stride, stride]
            if not isinstance(self.p, int):
                self.p = self.p[0]

        self.pretrained_feat_info = pretrained_feat_info
        self._feats = {}
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def build_vit(model_type: str) -> nn.Module:
        assert "dino" in model_type or "r3m" in model_type or "mvp" in model_type
        if "dinov2" in model_type:
            model = torch.hub.load("facebookresearch/dinov2", model_type)
        elif "dino" in model_type:
            model = torch.hub.load("facebookresearch/dino:main", model_type)
        elif "r3m" in model_type:
            model = load_r3m("resnet50")
        elif "mvp" in model_type:
            model_type = model_type.replace('mvp_', '')
            import mvp
            model = mvp.load(model_type)
        return model

    @staticmethod
    def build_conv(model_cfg, pretrained_feat_info) -> nn.Module:
        from simple_bc.encoder.spawnnet.vit_conv import ViTConv

        model = ViTConv(**model_cfg, pretrained_feat_info=pretrained_feat_info)
        return model

    @staticmethod
    def prune_model(model: nn.Module, layer: int = 9) -> nn.Module:
        """
        :param model: the model to prune
        :param layer: the layer to extract from. 0 is the first layer after the input
        :return: the pruned model
        """
        try:
            model.transformer.resblocks = model.transformer.resblocks[: layer + 1]
        except AttributeError:
            model.blocks = model.blocks[: layer + 1]
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """

        def interpolate_pos_encoding(
            self, x: torch.Tensor, w: int, h: int
        ) -> torch.Tensor:
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
            assert (
                w0 * h0 == npatch
            ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(
                    1, int(math.sqrt(N)), int(math.sqrt(N)), dim
                ).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            assert (
                int(w0) == patch_pos_embed.shape[-2]
                and int(h0) == patch_pos_embed.shape[-1]
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    def _get_hook(self, facet: str, layer: int):
        """
        generate a hook method for a specific block and facet.
        """
        feature_name = f"{facet}_{layer}"
        if facet in ["attn", "token"]:

            def _hook(model, input, output):
                self._feats[feature_name] = output

            return _hook

        if facet == "query":
            facet_idx = 0
        elif facet == "key":
            facet_idx = 1
        elif facet == "value":
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = (
                module.qkv(input)
                .reshape(B, N, 3, module.num_heads, C // module.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            self._feats[feature_name] = qkv[facet_idx]  # Bxhxtxd

        return _inner_hook

    def _register_hooks(self, feature_names) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        # feature names are in format of 'facet_layer'
        facets, layers = zip(*[name.split("_") for name in feature_names])
        layers = [int(layer) for layer in layers]
        facets_from_layers = {layer: facet for layer, facet in zip(layers, facets)}

        for block_idx, block in enumerate(self.vit.blocks):
            if block_idx in layers:
                facet = facets_from_layers[block_idx]
                if facet == "token":
                    self.hook_handlers.append(
                        block.register_forward_hook(self._get_hook(facet, block_idx))
                    )
                elif facet == "attn":
                    self.hook_handlers.append(
                        block.attn.attn_drop.register_forward_hook(
                            self._get_hook(facet, block_idx)
                        )
                    )
                elif facet in ["key", "query", "value"]:
                    self.hook_handlers.append(
                        block.attn.register_forward_hook(
                            self._get_hook(facet, block_idx)
                        )
                    )
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch: torch.Tensor, feature_names: List[str]) -> dict:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :param feature_names: A list of names of the features to extract. Should be in the format of "{faucet}_{layer}"
        :return : Dictionary of features
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = {}
        self._register_hooks(feature_names)

        _ = self.vit(batch)

        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (
            1 + (H - self.p) // self.stride[0],
            1 + (W - self.p) // self.stride[1],
        )
        return self._feats

    def _extract_clip_feature(self, batch: torch.Tensor):
        B, C, H, W = batch.shape
        self.num_patches = (H // self.p[0], W // self.p[1])
        return self.vit.forward(batch)

    def _extract_r3m_feature(self, batch: torch.Tensor, feature_names):
        # copied from the torchvision resnet package
        resnet_enc = self.vit.module.convnet
        x = resnet_enc.conv1(batch)
        x = resnet_enc.bn1(x)
        x = resnet_enc.relu(x)
        x = resnet_enc.maxpool(x)

        feat_dict = {}

        for i in range(1, 5):
            x = eval(f"resnet_enc.layer{i}")(x)

            if f"layer_{i}" in feature_names:
                feat = rearrange(x, "b c h w -> b (h w) c") + 1e-12 # norm errors
                feat = torch.cat([feat[:, 0:1], feat], dim=1)  # dummy cls token
                feat_dict[f"layer_{i}"] = feat

        return feat_dict

    def extract_descriptors(self, batch: torch.Tensor, feature_names) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        if "clip" in self.model_type:
            facet = "token"
            features = self._extract_clip_feature(batch)
            raise NotImplementedError
        elif "r3m" in self.model_type:
            features = self._extract_r3m_feature(batch, feature_names)
        else:
            features = self._extract_features(batch, feature_names)

        for name, val in features.items():
            # Token shape: * x (1 + pH x pW) x d
            val = val / val.norm(dim=-1, keepdim=True)
            features[name] = val
        return features

    def preprocess(self, obs):
        with torch.no_grad():
            ret = {}
            if "tokens" in obs:
                ret["tokens"], ret["cls"], ret["proprio"] = (
                    obs["tokens"],
                    obs["cls"],
                    obs["state"],
                )
                raise NotImplementedError
            else:
                if "rgb" in obs:
                    # normalize RGB image
                    rgb, ps = pack_one(obs["rgb"], "* c h w")  # [0, 255]
                    rgb = rgb.float() / 255.0
                    rgb = self.normalize(rgb)  # use imagenet mean and std
                    rgb = unpack_one(rgb, ps, "* c h w")
                    # debug
                    ret["video"] = rgb

            if "depth" in obs:
                depth, ps = pack_one(obs["depth"], "* c h w")
                depth = torch.clip(depth, 0.0, 2.0)
                depth = unpack_one(depth, ps, "* c h w")
                ret["depth"] = depth.float()

            if "state" in obs:
                ret["proprio"] = obs["state"]
        return ret

    def forward_vit(self, video, depth, pretrained_feats=None, **kwargs):
        """Video shape: (B, F, V, C, H, W)
        Conv feature shape: (B, F, V, D)
        """
        images, packed_shape = pack_one(video, "* c h w")
        depth, _ = pack_one(depth, "* c h w")
        rgbd = torch.cat([images, depth], dim=1)

        if pretrained_feats is None:
            with torch.no_grad():
                pretrained_feats = self.extract_descriptors(
                    images, self.pretrained_feat_info
                )  # * x (1 + pH x pW) x D
            if self.freeze_vit_to_random:
                for key, val in pretrained_feats.items():
                    pretrained_feats[key] = torch.zeros_like(val)
        conv_features, attn = self.conv.forward_feature(rgbd, pretrained_feats)
        conv_features = unpack_one(conv_features, packed_shape, "* c")
        attn = unpack_one(attn, packed_shape, "* g h w")
        return conv_features, attn

    def forward(self, obs, pretrained_feats=None, **kwargs):
        # V: Number of views
        # Feature: B x F x V x C
        processed_obs = self.preprocess(obs)
        feature, attn = self.forward_vit(processed_obs["video"], processed_obs["depth"], pretrained_feats, **kwargs)
        state = processed_obs["proprio"]  # B F D
        feature = rearrange(feature, "b f v c -> b f (v c)")
        feature = torch.cat([feature, state], dim=-1)
        feature = rearrange(feature, "b f d -> b (f d)")

        processed_obs["attn"] = attn

        return feature, processed_obs

    def get_pretrained_feats(self, obs):
        processed_obs = self.preprocess(obs)
        video = processed_obs["video"]
        depth = processed_obs["depth"]

        images, packed_shape = pack_one(video, "* c h w")

        with torch.no_grad():
            pretrained_feats = self.extract_descriptors(
                images, self.pretrained_feat_info
            )  # * x (1 + pH x pW) x D

        return pretrained_feats