from dataclasses import dataclass, field


@dataclass
class TransformConfig:
    s_rate: int = 128
    window: int = 4


@dataclass
class DataConfig:
    data_path: str = "/Users/rango/Desktop/GAIN Datasets/Sessions"
    label_type: str = "V"
    split_type: str = "loso"
    class_names: list[str] = field(default_factory=lambda: ["low", "high"])
    ground_truth_threshold: float = 4.5  # inclusive
    n_classes: int = 2


@dataclass
class TrainingConfig:
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.01
    num_epochs: int = 30
    log_dir: str = "logs/"
    overal_log_file: str = "logs.txt"


@dataclass
class EEGNetConfig:
    samples: int = 512
    dropout_rate: float = 0.5
    channels: int = 32


@dataclass
class TScepctionConfig:
    sampling_r: int = 128
    num_t: int = 15
    num_s: int = 15
    hidden: int = 32
    dropout_rate: float = 0.5
    input_size: tuple = (1, 32, 512)

@dataclass
class  DeepConvNetConfig:
    n_chan: int = 32
    n_time: int = 512
    dropout_rate: int = 0.5
    
@dataclass
class ShallowConvNetConfig:
    n_chan: int = 32
    n_time: int = 512
    dropout_rate: int = 0.5