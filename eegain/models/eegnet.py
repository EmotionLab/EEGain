import torch
import torch.nn as nn

from ._registry import register_model


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


@register_model
class EEGNet(nn.Module):
    def initial_block(self, dropout_rate):
        block1 = nn.Sequential(
            nn.Conv2d(
                1,
                self.F1,
                (1, self.kernelLength),
                stride=1,
                padding=(0, self.kernelLength // 2),
                bias=False,
            ),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            # Depth-wiseConv2D =======================
            Conv2dWithConstraint(
                self.F1,
                self.F1 * self.D,
                (self.channels, 1),
                max_norm=1,
                stride=1,
                padding=(0, 0),
                groups=self.F1,
                bias=False,
            ),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropout_rate),
        )

        block2 = nn.Sequential(
            # SeparableConv2D =======================
            nn.Conv2d(
                self.F1 * self.D,
                self.F1 * self.D,
                (1, self.kernelLength2),
                stride=1,
                padding=(0, self.kernelLength2 // 2),
                bias=False,
                groups=self.F1 * self.D,
            ),
            nn.Conv2d(
                self.F1 * self.D,
                self.F2,
                1,
                padding=(0, 0),
                groups=1,
                bias=False,
                stride=1,
            ),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout_rate),
        )
        return nn.Sequential(block1, block2)

    @staticmethod
    def classifier_block(input_size, n_classes):
        return nn.Sequential(
            nn.Linear(input_size, n_classes, bias=False), nn.Softmax(dim=1)
        )

    @staticmethod
    def calculate_out_size(model, channels, samples):
        """
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        """

        data = torch.rand(1, 1, channels, samples)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(
        self,
        n_classes,
        channels,
        dropout_rate,
        samples,
        kernel_length=64,
        kernel_length2=16,
        f1=8,
        d=2,
        f2=16,
    ):
        super(EEGNet, self).__init__()

        self.F1 = f1
        self.F2 = f2
        self.D = d
        self.samples = samples
        self.n_classes = n_classes
        self.channels = channels
        self.kernelLength = kernel_length
        self.kernelLength2 = kernel_length2
        self.dropoutRate = dropout_rate

        self.blocks = self.initial_block(dropout_rate)
        self.blockOutputSize = EEGNet.calculate_out_size(self.blocks, channels, samples)
        self.classifierBlock = EEGNet.classifier_block(
            self.F2 * self.blockOutputSize[1], n_classes
        )

    def forward(self, x):
        x = self.blocks(x)
        x = x.view(x.size()[0], -1)  # Flatten
        x = self.classifierBlock(x)

        return x
