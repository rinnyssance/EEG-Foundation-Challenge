import data_processing.data_loader as dl
import data_processing.data_split as ds

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

    print("Number of examples in each split minirelease")

    print(
        f"Train Set     : \t{len(train_set)}\n"
        f"Validation Set: \t{len(valid_set)}\n"
        f"Test set      : \t{len(test_Set)}\n"
    )

if __name__ == "__main__":
    main();
