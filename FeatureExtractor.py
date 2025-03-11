import numpy as np

class FeatureExtractor:

    def __init__(self, features, feature_thresholds, swap_intensity, logger=None):
        self.swap_intensity = swap_intensity

        self.logger = logger #what if None?

        self.features = features
        self.feature_thresholds = feature_thresholds

    def extract(self, geno, save_address=False): #extracts a dictionary based on a single solution
        schemas = {}

        for start in range(len(geno)):
            for end in range(len(geno)):
                if end >= start:
                    seq = geno[start:end+1]

                    # key = (tuple(delta) , tuple(cog)) # tuple of tuples [0] is delta, [1] is cog
                    key = tuple(tuple(self.features[i].get_feature_value(seq)) for i in range(len(self.features)))

                    if save_address:
                        val = (seq, (start, end+1))
                    else:
                        val = seq
                    if not key in schemas:
                        schemas[key] = set()
                    schemas[key].add(val)


        if self.swap_intensity >= 1:
            #TODO why three digits? we should ask features what are their lengths and default values, and prepare accordingly!
            empty_key = tuple((0,0,0) for i in range(len(self.features)))
            if not save_address:
                if empty_key in schemas:         
                    schemas[empty_key].update({""})
                else:
                    schemas[empty_key] = set([""])
            else:
                if empty_key in schemas:         
                    schemas[empty_key].update(set([("", (i,i)) for i in range(len(geno)+1)]))
                else:
                    schemas[empty_key] = set([("", (i,i)) for i in range(len(geno)+1)])

        self.logger.print_verbose(2, schemas)
        return schemas
