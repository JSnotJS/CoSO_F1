import random


class GenotypePool:
    def __init__(self, genotypes, sample_replacement=False):
        if len(genotypes) == 0:
            raise ValueError("Genotype pool is empty")

        self.genotypes = list(genotypes)
        self.sample_replacement = sample_replacement
        self._remaining = []
        self._reset_remaining()

    @classmethod
    def from_file(cls, path, sample_replacement=False):
        genotypes = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                genotype = line.strip()
                if genotype and not genotype.startswith("#"):
                    genotypes.append(genotype)
        return cls(genotypes, sample_replacement=sample_replacement)

    def _reset_remaining(self):
        self._remaining = self.genotypes[:]
        random.shuffle(self._remaining)

    def sample(self):
        if self.sample_replacement:
            return random.choice(self.genotypes)

        if not self._remaining:
            self._reset_remaining()
        return self._remaining.pop()

    def sample_many(self, count):
        if count < 0:
            raise ValueError("count must be non-negative")

        if self.sample_replacement:
            return [self.sample() for _ in range(count)]

        if count > len(self.genotypes):
            raise ValueError(
                "Cannot sample %d genotypes without replacement from a pool of %d genotypes"
                % (count, len(self.genotypes))
            )

        return [self.sample() for _ in range(count)]
