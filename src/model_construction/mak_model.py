import math
from braindecode.modules import LinearWithConstraint
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from braindecode.models import EEGNeX

class EEGNeXExtractor(EEGNeX):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        return x

BaseExtractor = EEGNeXExtractor(
    n_chans=129,
    n_outputs=129,
    n_times=200,
    sfreq=100
)

class MakConvModel(nn.Module): 
    def __init__(self,
        n_chans: int = 129,
        n_times: int = 300,
        mode: str = "pred",
        n_hidden_dims: int = 512,
        eeg_model: EEGNeXExtractor = BaseExtractor,
        activation: nn.Module = nn.ELU(),
        depth_multiplier: int = 2,
        filter_1: int = 8,
        filter_2: int = 32,
        avg_pool_block_4: int = 4,
        avg_pool_block_5: int = 16,
        kernel_block_1_2: int = 64,
        kernel_block_3_4: int = 32,
        kernel_block_5_6: int = 16,
        dilation_pool_block_4: int = 4,
        dilation_pool_block_5: int = 16,
        max_norm_linear: float = 0.25
    ):
        super().__init__(

        )

        self.n_times = n_times
        self.base_eeg_model = eeg_model
        self.filter_1 = filter_1
        self.filter_2 = filter_2
        self.filter_3 = self.filter_2 * depth_multiplier
        self.activation = activation
        self.kernel_block_1_2 = (1, kernel_block_1_2)
        self.kernel_block_3_4 = (1, kernel_block_3_4)
        self.kernel_block_5_6 = (1, kernel_block_5_6)
        self.avg_pool_block_4 = (1, avg_pool_block_4)
        self.avg_pool_block_5 = (1, avg_pool_block_5)
        self.dilation_pool_block_4 = (1, dilation_pool_block_4)
        self.dilation_pool_block_5 = (1, dilation_pool_block_5)
        self.channels = n_chans
        self.hidden_dims = n_hidden_dims
        self.output_dims = 1 if mode == "pred" else 10
        self.in_features = self._calculate_final_features()

        self.block_1 = nn.Sequential(
            self.base_eeg_model
        )

        self.block_2 = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_dims), self.activation,
            nn.Linear(self.hidden_dims, self.hidden_dims), self.activation,
            nn.Linear(self.hidden_dims, self.hidden_dims), self.activation,
            nn.Linear(self.hidden_dims, self.hidden_dims), self.activation,
            nn.Linear(self.hidden_dims, self.hidden_dims), self.activation,
            nn.Linear(self.hidden_dims, self.hidden_dims), self.activation,
            nn.Linear(self.hidden_dims, self.hidden_dims), self.activation,
            nn.Linear(self.hidden_dims, self.hidden_dims), self.activation,
            nn.Linear(self.hidden_dims, self.hidden_dims), self.activation,
        )

        self.final_layer = LinearWithConstraint(
            in_features=self.hidden_dims,
            out_features=self.output_dims,
            max_norm=max_norm_linear
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = torch.flatten(x, 1)
        x = self.block_2(x)
        x = self.final_layer(x)
        return x

    def _calculate_final_features(self) -> int: 
        p4 = self.avg_pool_block_4[1]
        p5 = self.avg_pool_block_5[1]

        pad4 = 1
        pad5 = 1

        T3 = math.floor((self.n_times + 2 * pad4 - p4) / p4) + 1
        T5 = math.floor((T3 + 2 * pad5 - p5) / p5) + 1

        final_features = self.filter_1 * T5

        return final_features




