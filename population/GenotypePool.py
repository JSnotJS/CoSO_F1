import random
from pathlib import Path


class GenotypePool:
    def __init__(self, genotypes, sample_replacement=False, fitness_by_genotype=None):
        if len(genotypes) == 0:
            raise ValueError("Genotype pool is empty")

        self.genotypes = list(genotypes)
        self.fitness_by_genotype = dict(fitness_by_genotype or {})
        self.sample_replacement = sample_replacement
        self._remaining = []
        self._reset_remaining()

    @classmethod
    def from_file(cls, path, sample_replacement=False):
        if Path(path).suffix.lower() == ".tsv":
            return cls.from_tsv(path, sample_replacement=sample_replacement)

        return cls.from_txt(path, sample_replacement=sample_replacement)

    @classmethod
    def from_txt(cls, path, sample_replacement=False):
        genotypes = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                genotype = line.strip()
                if genotype and not genotype.startswith("#"):
                    genotypes.append(genotype)
        return cls(genotypes, sample_replacement=sample_replacement)

    @classmethod
    def from_tsv(cls, path, sample_replacement=False):
        genotypes = []
        fitness_by_genotype = {}

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.rstrip("\r\n")
                if not line or line.startswith("#"):
                    continue

                fields = line.split("\t")
                if len(fields) < 2:
                    raise ValueError("%s:%d: expected genotype and fitness columns" % (path, line_no))

                genotype = fields[0].strip()
                if not genotype:
                    continue

                try:
                    fitness = float(fields[1].strip())
                except ValueError:
                    if line_no == 1:
                        continue
                    raise ValueError("%s:%d: invalid fitness value: %r" % (path, line_no, fields[1]))

                genotypes.append(genotype)
                fitness_by_genotype[genotype] = fitness

        return cls(genotypes, sample_replacement=sample_replacement, fitness_by_genotype=fitness_by_genotype)

    def get_fitness(self, genotype):
        return self.fitness_by_genotype.get(genotype)

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
