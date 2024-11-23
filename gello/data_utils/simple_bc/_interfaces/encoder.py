import torch
from abc import ABC, abstractmethod
import os


class Encoder(ABC, torch.nn.Module):
    def __init__(
        self,
        obs_shapes,  # dict of shapes of the observations.
        out_shape=None,  # Shape of the output of the encoder.
        num_frames=1,
        **kwargs,
    ):
        "Base class for all encoders."
        super().__init__()
        self.obs_shapes = obs_shapes
        self.out_shape = out_shape
        self.num_frames = num_frames

    @staticmethod
    def build_encoder(encoder_cfg):
        import simple_bc.encoder as e

        Encoder = eval(f"e.{encoder_cfg.name}")
        kwargs = dict(encoder_cfg)

        kwargs.pop("name")
        if "vit_cfg" in kwargs:
            kwargs.update(**kwargs.pop("vit_cfg"))
        encoder = Encoder(**kwargs)
        encoder.out_shape = eval(
            str(encoder.out_shape)
        )  # encoder shape may have already been evaluated in constructor
        return encoder

    def save(self, path):
        """
        Save the encoder's state dict to a file.
        """
        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load the encoder's state dict from a file.
        """
        self.load_state_dict(torch.load(path))

    @abstractmethod
    def preprocess(self, obs):
        """
        Preprocess the observation before passing it through the encoder.
        """
        pass

    @abstractmethod
    def forward(self, obs):
        """
        Forward pass of the encoder. Returns the encoded obs, which is used as input to the policy.
        """
        pass
