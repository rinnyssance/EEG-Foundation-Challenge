from typing import Any
from eegdash.hbn.windows import BaseConcatDataset
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from consts.model import (
    sub_rm,
    test_frac,
    valid_frac,
    seed
)

class DataSplitter:
    def __init__(self, dataset: DataFrame, window: BaseConcatDataset):
        self.__dataset = dataset
        self.__window = window
        self.__subjects: Any
        self.__train_subj = []
        self.__valid_subj = []
        self.__test_subj  = []
        self.__train_set: BaseConcatDataset;
        self.__valid_set: BaseConcatDataset;
        self.__test_set: BaseConcatDataset;

    def setUniqueData(self, prop: str) -> None:
        self.__subjects = self.__dataset["subject"].unique()

    def filterSubjects(self) -> None:
        self.__subjects = [s for s in self.__subjects if s not in sub_rm]

    def performSanityCheck(self):
        train_subj, valid_test_subj = train_test_split(
            self.__subjects,
            test_size=(valid_frac + test_frac),
            random_state=check_random_state(seed),
            shuffle=True
        )

        valid_subj, test_subj = train_test_split(
            valid_test_subj,
            test_size=test_frac,
            random_state=check_random_state(seed + 1),
            shuffle=True
        )

        assert (
            set(valid_subj) 
            | set(test_subj)
            | set(train_subj)
        ) == set(self.__subjects)

        self.__train_subj = train_subj
        self.__valid_subj = valid_subj
        self.__test_subj  = test_subj


    def splitToTrainTest(self):
        subject_split = self.__window.split("subject");
        train = []
        valid = []
        test  = []

        for s in subject_split:
            if s in self.__train_subj: train.append(subject_split[s])
            elif s in self.__valid_subj: valid.append(subject_split[s])
            elif s in self.__test_subj: test.append(subject_split[s])

        self.__train_set = BaseConcatDataset(train)
        self.__valid_set = BaseConcatDataset(valid)
        self.__test_set  = BaseConcatDataset(test)

    def getTrainSet(self) -> BaseConcatDataset:
        return self.__train_set;

    def getValidSet(self) -> BaseConcatDataset:
        return self.__valid_set;

    def getTestSet(self) -> BaseConcatDataset:
        return self.__test_set;
