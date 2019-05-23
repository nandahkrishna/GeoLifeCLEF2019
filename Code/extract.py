from environmental_raster_glc import PatchExtractor


class GeoLifeClefDataset():
    def __init__(self, extractor, dataset, labels):
        self.extractor = extractor
        self.labels = labels
        self.dataset = dataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tensor = self.extractor[self.dataset[idx]]
        return tensor, self.labels[idx]


if __name__ == '__main__':
    patch_extractor = PatchExtractor('/data/rasters_GLC19', size=64, verbose=True)

    patch_extractor.add_all()

    # example
    dataset_list = [(43.61, 3.88), (42.61, 4.88), (46.15, -1.1), (49.54, -1.7)]
    labels_list = [0, 1, 0, 1]

    dataset = GeoLifeClefDataset(patch_extractor, dataset_list, labels_list)

    print(len(dataset), ' elements in the dataset')
