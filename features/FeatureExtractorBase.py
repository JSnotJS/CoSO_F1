class FeatureExtractorBase:

    def __init__(self, features, feature_thresholds, swap_intensity, logger=None):
        self.swap_intensity = swap_intensity

        self.logger = logger #what if None?

        self.features = features
        self.feature_thresholds = feature_thresholds

    def extract(self, geno, save_address=False):
        ...
