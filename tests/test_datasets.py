import eegain.transforms as transforms
from eegain.data.datasets import DEAP, MAHNOB


def test_mahnob_subject_ids():
    mahnob_dataset = MAHNOB(
        "/home/raphael/Desktop/repos/GAIN_Biosignals/TSception/sessions/Sessions",
        label_type="A",
        transform=None,
    )
    subjects_ids = {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        11,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        29,
        30,
    }
    assert subjects_ids == set(mahnob_dataset.__get_subject_ids__())


def test_mahnob_subject():
    mahnob_dataset = MAHNOB(
        "/home/raphael/Desktop/repos/GAIN_Biosignals/TSception/sessions/Sessions",
        label_type="A",
        transform=None,
    )
    subject_data = mahnob_dataset.__get_subject__(1)[0]
    assert 20 == len(subject_data)


def test_mahnob_subject_session_shape():
    transform = transforms.Construct(
        [
            transforms.DropChannels(
                [
                    "EXG1",
                    "EXG2",
                    "EXG3",
                    "EXG4",
                    "EXG5",
                    "EXG6",
                    "EXG7",
                    "EXG8",
                    "GSR1",
                    "GSR2",
                    "Erg1",
                    "Erg2",
                    "Resp",
                    "Temp",
                    "Status",
                ]
            ),
        ]
    )

    mahnob_dataset = MAHNOB(
        "/home/raphael/Desktop/repos/GAIN_Biosignals/TSception/sessions/Sessions",
        label_type="A",
        transform=transform,
    )
    subject1_dict = mahnob_dataset.__get_subject__(1)[0]
    subject1 = subject1_dict[next(iter(subject1_dict))]

    assert 1 == subject1.shape[-3]
    assert 32 == subject1.shape[-2]
    assert 512 == subject1.shape[-1]


def test_deap_subject_ids():
    deap_dataset = DEAP(
        "/home/ttsmindashvili/Downloads/data_preprocessed_python",
        label_type="A",
        transform=None,
    )
    subjects_ids = {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        11,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        29,
        30,
        31,
        32,
    }
    assert subjects_ids == set(deap_dataset.__get_subject_ids__())
