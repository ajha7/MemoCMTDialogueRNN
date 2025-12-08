import torch
from torch.nn import CrossEntropyLoss as CELoss
from configs.base import Config


class CrossEntropyLoss(CELoss):
    """Rewrite CrossEntropyLoss to support init with kwargs"""

    def __init__(self, cfg: Config, **kwargs):
        # Get class weights from training data if available
        if hasattr(cfg, 'class_weights'):
            weights = torch.tensor(cfg.class_weights, dtype=torch.float32)
        else:
            # Default weights for IEMOCAP's typical imbalance
            # Adjust these based on your actual class distribution
            weights = torch.tensor([1.0, 2.0, 2.0, 1.5])  
        # super(CrossEntropyLoss, self).__init__(**kwargs)
        # self.cfg = cfg
        super().__init__(weight=weights, **kwargs)
        self.cfg = cfg

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # out = input[0]
        # return super().forward(out, target)
        return super().forward(input, target)
