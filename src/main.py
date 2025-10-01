from torch import nn
import data_processing.data_loader as dl
import data_processing.data_split as ds
import model_construction.base_model as bm

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
        .setModel(bm.DefaultModel)
        .setOptimizer(bm.DefaultOptimzer)
        .setLossFn(nn.MSELoss())
        .shouldPrintStats(True)
        .setEpoch(20)
        .buildModel()
    )
    model.trainModel()
    model.saveModel()

if __name__ == "__main__":
    main();
