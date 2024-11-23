import os
from abc import ABC, abstractmethod
import torch
from einops import rearrange


class Policy(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cached_actions = []

    def reset(self):
        self.cached_actions = []

    @staticmethod
    def build_policy(encoder_out_shape, policy_cfg, encoder_cfg):
        import simple_bc.policy as p

        Policy = eval(f"p.{policy_cfg.name}")
        kwargs = dict(policy_cfg)
        kwargs.pop("name")
        policy = Policy(obs_shape=encoder_out_shape, encoder_cfg=encoder_cfg, **kwargs)
        return policy

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
    def forward(self, *args, **kwargs):
        pass

    def act(self, *args, **kwargs):
        if len(self.cached_actions) == 0 or True:
            actions, info = self.forward(*args, **kwargs)
            assert len(actions) == 1  # Batch size should be 1
            self.cached_actions = actions[0].detach().cpu().numpy()
            self.cached_actions = self.cached_actions[:2]
        action, self.cached_actions = self.cached_actions[0], self.cached_actions[1:]
        return action, info
