import torch
from torch import nn

import eegain
import main
from eegain.data import EEGDataloader
from eegain.data.datasets import DREAMER
from eegain.logger import EmotionLogger
from eegain.models import TSception

# -------------- Preprocessing --------------
transform = eegain.transforms.Construct(
    [
        eegain.transforms.Filter(l_freq=0.3, h_freq=45),
        eegain.transforms.NotchFilter(freq=50),
        eegain.transforms.Resample(s_rate=128),
        eegain.transforms.Segment(duration=4, overlap=0),
    ]
)


# -------------- Dataset --------------
mahnob_dataset = DREAMER(
    "path/to/dreamer/mat/file",
    label_type="V",
    transform=transform,
)

# -------------- Dataloader --------------
eegloader = EEGDataloader(mahnob_dataset, batch_size=32).loso()


# -------------- Model --------------
model = TSception(
    num_classes=2,
    input_size=(1, 14, 512),
    sampling_r=128,
    num_t=15,
    num_s=15,
    hidden=32,
    dropout_rate=0.5,
    # pretrained=True
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
