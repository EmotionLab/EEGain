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
class DeepConvNet(nn.Module):
    @staticmethod
    def conv_block(in_f, out_f, dropout_rate, kernel_size, *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=dropout_rate),
            Conv2dWithConstraint(
                in_f, out_f, kernel_size, bias=False, max_norm=2, *args, **kwargs
            ),
            nn.BatchNorm2d(out_f),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3)),
        )

    @staticmethod
    def first_block(out_f, kernel_size, n_chan, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(
                1, out_f, kernel_size, padding=0, max_norm=2, *args, **kwargs
            ),
            Conv2dWithConstraint(
                25, 25, (n_chan, 1), padding=0, bias=False, max_norm=2
            ),
            nn.BatchNorm2d(out_f),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3)),
        )

    @staticmethod
    def last_block(in_f, out_f, kernel_size, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(
                in_f, out_f, kernel_size, max_norm=0.5, *args, **kwargs
            )
        )

    @staticmethod
    def calculate_out_size(model, n_chan, n_time):
        """
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        """

        data = torch.rand(1, 1, n_chan, n_time)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, channels, num_classes, dropout_rate, **kwargs):
        super(DeepConvNet, self).__init__()
        n_time = kwargs["sampling_r"]*kwargs["window"]
        # Please note that the kernel size in the original paper is (1, 10),
        # we found when the segment length is shorter than 4s (1s, 2s, 3s) larger kernel size will  cause network error.
        # Besides using (1, 5) when EEG segment is 4s gives slightly higher ACC and F1 with a smaller model size.
        kernel_size = (1, 5)
        n_filt_first_layer = 25
        n_filt_later_layer = [25, 50, 100, 200]

        first_layer = DeepConvNet.first_block(n_filt_first_layer, kernel_size, channels)
        middle_layers = nn.Sequential(
            *[
                DeepConvNet.conv_block(inF, outF, dropout_rate, kernel_size)
                for inF, outF in zip(n_filt_later_layer, n_filt_later_layer[1:])
            ]
        )

        self.allButLastLayers = nn.Sequential(first_layer, middle_layers)

        self.fSize = DeepConvNet.calculate_out_size(
            self.allButLastLayers, channels, n_time
        )
        self.lastLayer = DeepConvNet.last_block(
            n_filt_later_layer[-1], num_classes, (1, self.fSize[1])
        )

    def forward(self, x):
        x = self.allButLastLayers(x)
        x = self.lastLayer(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x
