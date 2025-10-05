from torch import nn
from torch.optim import AdamW
import data_processing.data_loader as dl
import data_processing.data_split as ds
from model_construction import mak_model
import model_construction.base_model as bm
import consts.model as cm
from model_construction.mak_model import (
    MakConvModel
)

test_mak_model = MakConvModel(
    n_chans=129,
    n_times=400,
    mode="pred"
)

test_optimizer = AdamW(
    test_mak_model.parameters(),
    lr=cm.lr,
    weight_decay=cm.weight_decay
)

def main():
    loader = dl.DataLoader()
    loader.createDataDir()
    loader.loadDataFromDataset()
    loader.preprocessDataset()
    loader.filterDataSet()

    dataset = loader.getFilteredDataset()
    window = dl.createWindowFromEvents(dataset)

    dl.addColumns(
        dataset,
        window,
        "target",
        "rt_from_stimulus",
        "rt_from_trialstart",
        "stimulus_onset",
        "response_onset",
        "correct",
        "response_type"
    )

    metadata = window.get_metadata()
    dl.inspectMetadata(metadata)

    splitter = ds.DataSplitter(metadata, window)
    splitter.setUniqueData("subject")
    splitter.filterSubjects()
    splitter.performSanityCheck()
    splitter.splitToTrainTest()

    train_set = splitter.getTrainSet()
    valid_set = splitter.getValidSet()
    test_Set  = splitter.getTestSet()

    print(
         "\n\nNumber of examples in each split minirelease\n"
        f"Train Set     : \t{len(train_set)}\n"
        f"Validation Set: \t{len(valid_set)}\n"
        f"Test set      : \t{len(test_Set)}\n"
    )

    modelBuilder = bm.BaseModelBuilder()
    model = (
        modelBuilder
        .setTrainSet(train_set)
        .setValidationSet(valid_set)
        .setTestSet(test_Set)
        .setModel(test_mak_model)
        .setOptimizer(test_optimizer)
        .setLossFn(nn.MSELoss())
        .shouldPrintStats(True)
        .setEpoch(20)
        .buildModel()
    )
    model.trainModel()
    model.saveModel("mak_conv_model")

if __name__ == "__main__":
    main();
