import torch
from torch import nn

import eegain
import main
from eegain.data import EEGDataloader
from eegain.data.datasets import SeedIV
from eegain.logger import EmotionLogger
from eegain.models import TSception

# -------------- Preprocessing --------------
channels_to_drop_seed_iv = [
    "FPZ",
    "F5",
    "F1",
    "F2",
    "F6",
    "FT7",
    "FC3",
    "FCZ",
    "FC4",
    "FT8",
    "C5",
    "C1",
    "C2",
    "C6",
    "TP7",
    "CP3",
    "CPZ",
    "CP4",
    "TP8",
    "P5",
    "P1",
    "PZ",
    "P2",
    "P6",
    "PO7",
    "PO5",
    "PO6",
    "PO8",
    "CB1",
    "CB2",
]

transform = eegain.transforms.Construct(
    [
        eegain.transforms.DropChannels(channels_to_drop_seed_iv),
        eegain.transforms.Filter(l_freq=0.3, h_freq=45),
        eegain.transforms.NotchFilter(freq=50),
        eegain.transforms.Resample(s_rate=128),
        eegain.transforms.Segment(duration=4, overlap=0),
    ]
)

# -------------- Dataset --------------
seed_dataset = SeedIV(
    "path/to/seed/folder",
    label_type="V",
    transform=transform,
)

# -------------- Dataloader --------------
eegloader = EEGDataloader(seed_dataset, batch_size=32).loso()

# -------------- Model --------------
model = TSception(
    num_classes=4,
    input_size=(1, 32, 512),
    sampling_r=128,
    num_t=15,
    num_s=15,
    hidden=32,
    dropout_rate=0.5,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
loss_fn = nn.CrossEntropyLoss()

logger = EmotionLogger(log_dir="logs/")

# -------------- Training --------------
for loader in eegloader:
    main.run(
        model=model,
        train_dataloader=loader["train"],
        test_dataloader=loader["test"],
        test_subject_ids=loader["test_subject_indexes"],
        optimizer=optimizer,
        loss_fn=loss_fn,
        epoch=100,
    )
