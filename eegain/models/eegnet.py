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
                (1, self.kernelLength1),
                stride=1,
                padding=(0, self.kernelLength1 // 2),
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
            nn.AvgPool2d((1, self.pool_size1), stride=self.pool_size1), # [NEW] Using scaled pool size
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
            nn.AvgPool2d((1, self.pool_size2), stride=self.pool_size2), # [NEW] Using scaled pool size
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
        num_classes,
        channels,
        dropout_rate,
        kernel_length1=64,
        kernel_length2=16,
        f1=8,
        d=2,
        f2=16,
        **kwargs
    ):
        super(EEGNet, self).__init__()
        samples = kwargs["sampling_r"]*kwargs["window"]
        self.F1 = f1
        self.F2 = f2
        self.D = d
        self.samples = samples
        self.n_classes = num_classes
        self.channels = channels
        
        # [NEW] Scale kernel length and pooling sizes based on sampling rate
        sampling_rate = kwargs.get("sampling_r", 128)  # Default to 128Hz
        scaling_factor = sampling_rate / 128  # Calculate scaling factor
        
        self.kernelLength1 = int(kernel_length1 * scaling_factor)
        self.kernelLength2 = int(kernel_length2 * scaling_factor)
        self.pool_size1 = int(4 * scaling_factor)  # [NEW] Scale pool size 1
        self.pool_size2 = int(8 * scaling_factor)  # [NEW] Scale pool size 2
        self.dropoutRate = dropout_rate

        self.blocks = self.initial_block(dropout_rate)
        
        # [NEW] Set classifier to None and create it dynamically in the first forward pass
        self.classifierBlock = None
    
    def forward(self, x):
        x = self.blocks(x)
        x = x.view(x.size()[0], -1)  # Flatten
        
        # [NEW] Create the classifier on the first forward pass with correct size
        # and move to the same device as the input tensor
        if self.classifierBlock is None:
            input_size = x.size(1)
            device = x.device
            print(f"[NEW] Creating classifier with input_size={input_size} on device {device}")
            self.classifierBlock = EEGNet.classifier_block(input_size, self.n_classes).to(device)
            
        x = self.classifierBlock(x)
        return x