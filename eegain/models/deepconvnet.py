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
        
        # [NEW] Scale kernel size based on sampling rate
        sampling_rate = kwargs.get("sampling_r", 128)  # Default to 128Hz
        scaling_factor = sampling_rate / 128  # Calculate scaling factor
        
        # [NEW] Base kernel size of 5 at 128Hz, scale accordingly
        kernel_size_base = 5
        kernel_size_scaled = int(kernel_size_base * scaling_factor)
        kernel_size = (1, kernel_size_scaled)
        print(f"[NEW] Using kernel size {kernel_size} (scaling_factor={scaling_factor})")
        
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
        
        # [NEW] Set up to create lastLayer dynamically in first forward pass
        self.n_filt_later_layer = n_filt_later_layer
        self.num_classes = num_classes
        self.lastLayer = None

    def forward(self, x):
        x = self.allButLastLayers(x)
        
        # [NEW] Create last layer dynamically based on actual feature size
        if self.lastLayer is None:
            feature_size = x.size()
            print(f"[NEW] Feature size before last layer: {feature_size}")
            # Ensure kernel size doesn't exceed feature map size
            kernel_size = (1, feature_size[3])
            self.lastLayer = DeepConvNet.last_block(
                self.n_filt_later_layer[-1], self.num_classes, kernel_size
            ).to(x.device)
            
        x = self.lastLayer(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x