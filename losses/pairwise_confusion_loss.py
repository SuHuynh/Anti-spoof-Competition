import torch
import torch.nn as nn

class PairwiseConfusion(nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self):
        super(PairwiseConfusion, self).__init__()

    def check_type_forward(self, in_types):
        assert len(in_types) == 1

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x):
        # self.check_type_forward(x)

        batch_size = x.size(0)
        # print('===================', batch_size)
        if float(batch_size) % 2 != 0:
            raise Exception('Incorrect batch size provided')
        batch_left = x[:int(0.5*batch_size)]
        batch_right = x[int(0.5*batch_size):]
        loss  = torch.norm((batch_left - batch_right).abs(),2, 1).sum() / float(batch_size/2)
        return loss