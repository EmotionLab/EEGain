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
class ShallowConvNet(nn.Module):
    @staticmethod
    def first_block(out_f, kernel_size, n_chan, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(
                1, out_f, kernel_size, padding=0, max_norm=2, *args, **kwargs
            ),
            Conv2dWithConstraint(
                40, 40, (n_chan, 1), padding=0, bias=False, max_norm=2
            ),
            nn.BatchNorm2d(out_f),
        )

    def __init__(self, channels, num_classes, dropout_rate, **kwargs):
        super(ShallowConvNet, self).__init__()
        n_time = kwargs["sampling_r"]*kwargs["window"]
        
        # [NEW] Scale kernel size and avgpool size based on sampling rate
        sampling_rate = kwargs.get("sampling_r", 128)  # Default to 128Hz
        scaling_factor = sampling_rate / 128  # Calculate scaling factor
        
        # [NEW] Scale kernel size and pooling size
        kernel_size_base = 25
        pool_size_base = 75
        pool_stride_base = 15
        
        kernel_size_scaled = int(kernel_size_base * scaling_factor)
        pool_size_scaled = int(pool_size_base * scaling_factor)
        pool_stride_scaled = int(pool_stride_base * scaling_factor)
        
        kernel_size = (1, kernel_size_scaled)
        n_filt_first_layer = 40

        print(f"[NEW] Using kernel size {kernel_size} (scaling_factor={scaling_factor})")
        print(f"[NEW] Using pool size {pool_size_scaled} with stride {pool_stride_scaled}")
        
        self.firstLayer = ShallowConvNet.first_block(
            n_filt_first_layer, kernel_size, channels
        )
        self.avgpool = nn.AvgPool2d((1, pool_size_scaled), stride=(1, pool_stride_scaled))
        self.dp = nn.Dropout(p=dropout_rate)
        self.num_classes = num_classes
        
        # [NEW] Set last layer to None, will create dynamically in forward pass
        self.lastLayer = None

    def forward(self, x):
        x = self.firstLayer(x)
        x = torch.log(self.avgpool(x.pow(2)))
        x = self.dp(x)
        x = x.view(x.size()[0], -1)
        
        # [NEW] Create last layer dynamically based on actual feature size
        if self.lastLayer is None:
            in_features = x.size(1)
            device = x.device
            print(f"[NEW] Creating last layer with in_features={in_features}, device={device}")
            self.lastLayer = nn.Linear(in_features, self.num_classes).to(device)
            
        x = self.lastLayer(x)
        return x