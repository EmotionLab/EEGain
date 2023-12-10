import logging

import torch
import torch.nn as nn

from ._registry import register_model
from .cnnadapters import ChapterAdapter, RebuffiAdapter

logger = logging.getLogger("Model")


class ZeroModule(nn.Module):
    def __init__(self):
        super(ZeroModule, self).__init__()
    def forward(self,x):
        return torch.tensor(0.)
                                        
class TSception_Conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 pool,
                 add_rebuffi_before = False,
                 add_rebuffi_after  = False,
                 add_chapter = False,
                 adapter_compression_rate = 0):
        
        super(TSception_Conv, self).__init__()

        log_info = "\n--".join(
            [f"{n}={v}" for n, v in locals().items() if n not in ["self", "__class__"]]
        )
        logger.info(f"Using model: \n{self.__class__.__name__}(\n--{log_info})")
        
        self.conv = nn.Conv2d(in_channels  = in_channels,
                              out_channels = out_channels,
                              kernel_size  = kernel_size,
                              stride       = stride)
        
        self.chapter_adapter   = ChapterAdapter(in_channels  = in_channels,
                                                out_channels = out_channels,
                                                kernel_size  = kernel_size,
                                                stride       = stride,
                                                compression_rate = adapter_compression_rate) if ((adapter_compression_rate > 0) and add_chapter) else ZeroModule()
        
        self.rebuffi_adapter_1 = RebuffiAdapter(in_channels  = out_channels,
                                                out_channels = out_channels,
                                                stride       = stride) if add_rebuffi_before else nn.Identity()
        
        self.rebuffi_adapter_2 = RebuffiAdapter(in_channels  = out_channels,
                                                out_channels = out_channels,
                                                stride       = stride) if add_rebuffi_after  else nn.Identity()        

        self.activation_fn = nn.LeakyReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool))
    
    def forward(self,x):
        out = self.conv(x)
        out = self.rebuffi_adapter_1(out)
        delta_out = self.chapter_adapter(x)
        out += delta_out
        out = self.activation_fn(out)
        out = self.rebuffi_adapter_2(out)
        return self.avg_pool(out)
        
@register_model
class AdaptiveTSception(nn.Module):
    def __init__(self,
                 num_classes,
                 input_size,
                 sampling_r,
                 num_t,
                 num_s,
                 hidden,
                 dropout_rate,
                 add_bitfit = True,
                 add_rebuffi_before = True,
                 add_rebuffi_after  = True,
                 add_chapter = True,
                 adapter_compression_rate = 1):
        # input_size: 1 x EEG channel x datapoint
        super(AdaptiveTSception, self).__init__()
        self.add_bitfit = add_bitfit
        
        self.compression_rate = adapter_compression_rate
        self.inception_window = [0.25, 0.125, 0.0625]
        self.pool = 8
        # by setting the convolutional kernel being (1, length) and the strides being 1 we can use conv 2d to
        # achieve the 1d convolution operation
        
        # TEMPORAL:
        
        self.tsception1 = TSception_Conv(in_channels  = 1,
                                         out_channels = num_t,
                                         kernel_size  = (1, int(self.inception_window[0] * sampling_r)),
                                         stride       = 1,
                                         pool         = self.pool,
                                         add_rebuffi_before = add_rebuffi_before,
                                         add_rebuffi_after  = add_rebuffi_after,
                                         add_chapter        = add_chapter,
                                         adapter_compression_rate = adapter_compression_rate)
        
        self.tsception2 = TSception_Conv(in_channels  = 1,
                                         out_channels = num_t,
                                         kernel_size  = (1, int(self.inception_window[1] * sampling_r)),
                                         stride       = 1,
                                         pool         = self.pool,
                                         add_rebuffi_before = add_rebuffi_before,
                                         add_rebuffi_after  = add_rebuffi_after,
                                         add_chapter        = add_chapter,
                                         adapter_compression_rate = adapter_compression_rate)
        
        self.tsception3 = TSception_Conv(in_channels  = 1,
                                         out_channels = num_t,
                                         kernel_size  = (1, int(self.inception_window[2] * sampling_r)),
                                         stride       = 1,
                                         pool         = self.pool,
                                         add_rebuffi_before = add_rebuffi_before,
                                         add_rebuffi_after  = add_rebuffi_after,
                                         add_chapter        = add_chapter,
                                         adapter_compression_rate = adapter_compression_rate)
        
        # ================================================================
        # SPATIAL:

        self.sception1 = TSception_Conv(in_channels  = num_t,
                                        out_channels = num_s,
                                        kernel_size  = (int(input_size[1]), 1),
                                        stride       = 1,
                                        pool         = int(self.pool * 0.25),
                                        add_rebuffi_before = add_rebuffi_before,
                                        add_rebuffi_after  = add_rebuffi_after,
                                        add_chapter        = add_chapter,
                                        adapter_compression_rate = adapter_compression_rate)

        self.sception2 = TSception_Conv(in_channels  = num_t,
                                        out_channels = num_s,
                                        kernel_size  = (int(input_size[1] * 0.5), 1),
                                        stride       = (int(input_size[1] * 0.5), 1),
                                        pool         = int(self.pool * 0.25),
                                        add_rebuffi_before = add_rebuffi_before,
                                        add_rebuffi_after  = add_rebuffi_after,
                                        add_chapter        = add_chapter,
                                        adapter_compression_rate = adapter_compression_rate)
        
        # ================================================================
        # FUSION:

        self.fusion_layer = TSception_Conv(in_channels  = num_s,
                                           out_channels = num_s,
                                           kernel_size  = (3, 1),
                                           stride       = 1,
                                           pool         = 4,
                                           add_rebuffi_before = add_rebuffi_before,
                                           add_rebuffi_after  = add_rebuffi_after,
                                           add_chapter        = add_chapter,
                                           adapter_compression_rate = adapter_compression_rate)
        
        # ================================================================
        
        self.BN_t      = nn.BatchNorm2d(num_t)
        self.BN_s      = nn.BatchNorm2d(num_s)
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
    
    def freeze(self):
        layers = [self.tsception1.conv, self.tsception2.conv, self.tsception3.conv,
                  self.sception1.conv,  self.sception2.conv,  self.fusion_layer.conv,
                  self.BN_t,            self.BN_s,            self.BN_fusion,
                  self.fc[0],           self.fc[-1]]
        for layer in layers:
            layer.weight.requires_grad = False
            if not self.add_bitfit:
                layer.bias.requires_grad = False

    def load_missing_parametes(self, state_dict):
        self.tsception1.conv.weight.data   = state_dict['tsception1.0.weight']
        self.tsception2.conv.weight.data   = state_dict['tsception2.0.weight']
        self.tsception3.conv.weight.data   = state_dict['tsception3.0.weight']

        self.sception1.conv.weight.data    = state_dict['sception1.0.weight']
        self.sception2.conv.weight.data    = state_dict['sception2.0.weight']

        self.fusion_layer.conv.weight.data = state_dict['fusion_layer.0.weight']

        self.tsception1.conv.bias.data     = state_dict['tsception1.0.bias']
        self.tsception2.conv.bias.data     = state_dict['tsception2.0.bias']
        self.tsception3.conv.bias.data     = state_dict['tsception3.0.bias']

        self.sception1.conv.bias.data      = state_dict['sception1.0.bias']
        self.sception2.conv.bias.data      = state_dict['sception2.0.bias']

        self.fusion_layer.conv.bias.data   = state_dict['fusion_layer.0.bias'] 