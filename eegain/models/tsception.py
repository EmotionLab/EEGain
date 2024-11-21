import logging

import torch
import torch.nn as nn

from ._registry import register_model

logger = logging.getLogger("Model")


@register_model
class TSception(nn.Module):
    @staticmethod
    def conv_block(in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=kernel,
                stride=step,
            ),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)),
        )

    def __init__(
        self, num_classes, input_size, sampling_r, num_t, num_s, hidden, dropout_rate, **kwargs
    ):
        # input_size: 1 x EEG channel x datapoint
        super(TSception, self).__init__()

        log_info = "\n--".join(
            [f"{n}={v}" for n, v in locals().items() if n not in ["self", "__class__"]]
        )
        logger.info(f"Using model: \n{self.__class__.__name__}(\n--{log_info})")

        self.inception_window = [0.25, 0.125, 0.0625]
        self.pool = 8
        # by setting the convolutional kernel being (1, length) and the strides being 1 we can use conv 2d to
        # achieve the 1d convolution operation
        self.tsception1 = TSception.conv_block(
            1, num_t, (1, int(self.inception_window[0] * sampling_r)), 1, self.pool
        )
        self.tsception2 = TSception.conv_block(
            1, num_t, (1, int(self.inception_window[1] * sampling_r)), 1, self.pool
        )
        self.tsception3 = TSception.conv_block(
            1, num_t, (1, int(self.inception_window[2] * sampling_r)), 1, self.pool
        )

        self.sception1 = TSception.conv_block(
            num_t, num_s, (int(input_size[1]), 1), 1, int(self.pool * 0.25)
        )
        self.sception2 = TSception.conv_block(
            num_t,
            num_s,
            (int(input_size[1] * 0.5), 1),
            (int(input_size[1] * 0.5), 1),
            int(self.pool * 0.25),
        )

        self.fusion_layer = TSception.conv_block(num_s, num_s, (3, 1), 1, 4)
        self.BN_t = nn.BatchNorm2d(num_t)
        self.BN_s = nn.BatchNorm2d(num_s)
        self.BN_fusion = nn.BatchNorm2d(num_s)

        self.fc = nn.Sequential(
            nn.Linear(num_s, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        y = self.tsception1(x)
        out = y
        y = self.tsception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.tsception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.sception1(out)
        out_ = z
        z = self.sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)

        return out
