import logging
from typing import List

import mne

logger = logging.getLogger("Preprocessing")


class Construct:
    """
    Construct several transforms together

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        transform = eegain.transforms.Construct([
            transforms.Crop(t_min=30, t_max=-30),
            transforms.DropChannels(['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8',
                                    'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Temp', 'Status']),
            transforms.Filter(l_freq=0.3, h_freq=45),
            transforms.NotchFilter(freq=50),
            transforms.Resample(sampling_r=128)
        ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

        log_preprocess = "\n--".join([str(t) for t in self.transforms])
        logger.info(f"Using preprocessing:\n--{log_preprocess}")

    def __call__(self, data) -> mne.epochs.Epochs:
        data_ = None
        for t in self.transforms:
            if data_:
                data = data_
            data_ = t(data)
        return data if data_ is None else data_

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class Crop:
    """
    Crop the signal based on t_min and t_max

    note:
        if t_max is negative it crops last part of the signal
        if t_max is None, it keeps all data after t_min
    """

    def __init__(self, t_min: float, t_max=None) -> None:
        self.t_min = t_min
        self.t_max = t_max

    def __call__(self, data) -> None:
        if self.t_max is None:
            # Use the entire signal duration if t_max is None
            t_max = data.tmax
        else:
            # Original logic for non-None values
            t_max = self.t_max if self.t_max > 0 else data.tmax + self.t_max
            
        data.crop(tmin=self.t_min, tmax=t_max, verbose=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(t_min={self.t_min}, t_max={self.t_max})"


class Filter:
    """
    Filter signal based on low and high frequency
    """

    def __init__(self, l_freq: float, h_freq: float) -> None:
        self.l_freq = l_freq
        self.h_freq = h_freq

    def __call__(self, data) -> None:
        iir_params = dict(order=8, ftype="butter")
        data.filter(l_freq=self.l_freq, h_freq=self.h_freq, picks=None, method='iir', iir_params=iir_params, verbose=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(l_freq={self.l_freq}, h_freq={self.h_freq})"


class NotchFilter:
    """
    Filter signal based using Notch filter. it drops specific frequency
    """

    def __init__(self, freq: float) -> None:
        self.freq = freq

    def __call__(self, data) -> None:
        data.notch_filter(freqs=self.freq, verbose=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(freq={self.freq})"


class DropChannels:
    """
    Drops unnecessary channels fom the data
    """

    def __init__(self, channels: List[str]) -> None:
        self.channels = channels

    def __call__(self, data) -> None:
        data.drop_channels(self.channels)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(channels={self.channels})"


class Resample:
    """
    Resample the signal based on target frequency
    """

    def __init__(self, sampling_r: float) -> None:
        self.sampling_r = sampling_r

    def __call__(self, data) -> None:
        data.resample(self.sampling_r, verbose=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sampling_r={self.sampling_r})"


class Segment:
    """
    Creates the segments/windows based on the required duration
    """

    def __init__(self, duration: float, overlap: float) -> None:
        self.duration = duration
        self.overlap = overlap

    def __call__(self, data) -> mne.epochs.Epochs:
        segments = mne.make_fixed_length_epochs(
            data,
            duration=self.duration,
            overlap=self.overlap,
            verbose=False,
            preload=True,
        )
        return segments

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(duration={self.duration}, overlap={self.overlap})"
