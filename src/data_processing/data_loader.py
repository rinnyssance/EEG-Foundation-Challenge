from pathlib import Path
from pandas import DataFrame
import torch
from eegdash.dataset import EEGChallengeDataset
from eegdash.hbn.windows import (
    BaseConcatDataset,
    add_aux_anchors,
    add_extras_columns,
    annotate_trials_with_target,
    keep_only_recordings_with
)
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_windows_from_events
)

import consts.data

class DataLoader:

    def __init__(self) -> None:
        self.isGpuPresent()
        self.__path = Path("data")

    __dataset: EEGChallengeDataset;
    __path: Path;

    def isGpuPresent(self):
        msg = "CUDA-enabled GPU found. Training should be faster"
        if not torch.cuda.is_available():
            msg = (
                "No GPU found. Training will be carried out on the CPU, which might be "
                "slower.\n\nIf running on Google Colab, you can request a GPU run time by"
                " clicking\n `Runtime/Change runtime type` in the top bar menu, then "
                "selecting 'T4 GPU'\nunder 'Hardware accelarator'"
            )
        print(msg)


    def createDataDir(self, name: str = "data") -> None:
        if not self.__path:
            self.__path = Path(name)
        self.__path.mkdir(parents=True, exist_ok=True)
        print("[DEBUG] Successfully created data dir")


    def loadDataFromDataset(self) -> None:
        dataset = EEGChallengeDataset(
            task="contrastChangeDetection",
            release="R5",
            cache_dir=self.__path,
            mini=True
        )
        self.__dataset = dataset
        print("[DEBUG] Successfully loaded dataset!")


    def preprocessDataset(self) -> None:
        offline_transformation = [
            Preprocessor(
                annotate_trials_with_target,
                target_field="rt_from_stimulus",
                epoch_length = consts.data.EPOCH_LEN,
                require_stimulus=True,
                require_response=True,
                apply_on_array=False
            ),
            Preprocessor(add_aux_anchors, apply_on_array=False)
        ]
        preprocess(self.__dataset, offline_transformation, n_jobs=1)


    def filterDataSet(self) -> None:
        param: str = consts.data.ANCHOR
        self.__filtered_dataset = keep_only_recordings_with(param, self.__dataset)


    def getUnFilteredDataset(self) -> EEGChallengeDataset:
        return self.__dataset

    def getFilteredDataset(self) -> BaseConcatDataset:
        return self.__filtered_dataset


def createWindowFromEvents(dataset: BaseConcatDataset) -> BaseConcatDataset:
    single_window = create_windows_from_events(
        dataset,
        mapping = {consts.data.ANCHOR: 0},
        trial_start_offset_samples=int(consts.data.SHIFT_AFTER_STIM * 
                                      consts.data.SFREQ),
        trial_stop_offset_samples=int((consts.data.SHIFT_AFTER_STIM + consts.data.WINDOW_LEN) * 
                                      consts.data.SFREQ),
        window_size_samples=int(consts.data.EPOCH_LEN * 
                                consts.data.SFREQ),
        window_stride_samples=consts.data.SFREQ,
        preload=True
    )

    return single_window


def addColumns(dataset: BaseConcatDataset,
              windows: BaseConcatDataset, 
              *keys: str) -> None:
    add_extras_columns(
        windows,
        dataset,
        desc = consts.data.ANCHOR,
        keys=keys
    )


def setEpochLenAndFreq(epoch_len: float = 2.0, s_freq: int = 100):
    print(
        f"[DEBUG] set epoch and sfreq\n"
         "\tEpoch len: {epoch_len}\n\tS_Freq: {s_freq}"
    )
    return (epoch_len, s_freq)


def inspectMetadata(data: DataFrame) -> None:
    from matplotlib.pylab import plt
    fig, ax = plt.subplots(figsize=(15, 5))
    ax = data['target'].plot.hist(bins=30, ax=ax, color='lightblue')
    ax.set_xlabel("Response Time (s)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Response Times")
    plt.savefig("response_time_distribution.png")
