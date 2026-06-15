import numpy as np

from features.FeatureExtractorBase import FeatureExtractorBase
from features.f1.F1Tree import F1Parser, collect_subtrees, serialize_f1


class FeatureExtractor(FeatureExtractorBase):

    def __init__(self, features, feature_thresholds, swap_intensity, logger=None):
        super().__init__(features, feature_thresholds, swap_intensity, logger)

    def _empty_feature_tuple(self):
        vals = []
        for feature in self.features:
            fv = feature.get_feature_value("")
            vals.append(tuple(float(x) for x in np.asarray(fv).tolist()))
        return tuple(vals)

    def _feature_key_for_seq(self, seq):
        return tuple(tuple(self.features[i].get_feature_value(seq)) for i in range(len(self.features)))

    def extract(self, geno, save_address=False):
        schemas = {}

        tree = F1Parser(geno).parse()

        if tree is None:
            key = self._feature_key_for_seq(geno)
            if save_address:
                schemas[key] = {(geno, (0, len(geno)))}
            else:
                schemas[key] = {geno}
        else:
            for node in collect_subtrees(tree):
                seq = serialize_f1(node)
                key = self._feature_key_for_seq(seq)

                if save_address:
                    val = (seq, (node.start, node.end))
                else:
                    val = seq

                if key not in schemas:
                    schemas[key] = set()
                schemas[key].add(val)

        if self.swap_intensity >= 1:
            empty_key = self._empty_feature_tuple()
            if not save_address:
                if empty_key in schemas:
                    schemas[empty_key].update({""})
                else:
                    schemas[empty_key] = {""}
            else:
                empty_entries = {("", (i, i)) for i in range(len(geno) + 1)}
                if empty_key in schemas:
                    schemas[empty_key].update(empty_entries)
                else:
                    schemas[empty_key] = empty_entries

        if self.logger is not None:
            self.logger.print_verbose(2, schemas)
        return schemas
