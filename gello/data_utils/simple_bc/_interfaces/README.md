`Encoder`s are responsible for handling raw input.

Classes should be entire algorithms, such as `impala.py` or `ddt.py`. Because `Encoder`s are fully responsible for processing the input, they also must preprocess (i.e. transformations, etc.) the output of the `Data` class.

Methods:
```
forward(self, obs):

loss(self, obs, act):

update(self, obs, act):

save(self, path):

load(self, path):

_build_network(self, nn_cfg, device):
```

TODO: add typing hints and more documentation to this page.