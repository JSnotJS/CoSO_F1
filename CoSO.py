import bisect
import random
import numpy.random
import argparse
from CoSOArchive import CoSOArchive
from FramsProblem_VertposF1 import FramsProblem
from features.f1.FeatureExtractor import FeatureExtractor
from features.f1.FeatureDeltaSpatial import FeatureDeltaSpatial
from features.f1.FeatureDeltaDirection import FeatureDeltaDirection
from population.GenotypePool import GenotypePool
from utils import Logger, ExperimentLogger
import time

class CoSO:

    def __init__(self):
        args = self.parse_args()

        if args.seed == None:
            args.seed = random.randint(0, 100000)
        random.seed(args.seed)
        numpy.random.seed(args.seed)

        self.frams = FramsProblem(
            frams_path=args.frams_path,
            context_path="D:/STUDIA_v2/THEIR_MAGISTRY/CoSO/context/",
            task="vertpos",
            eval_increment_fun=self.increment_eval_counter
        )

        self.swap_intensity = args.swap_intensity
        self.verbose = args.verbose

        self.thresh_delta = args.thresh_delta
        self.thresh_cog = args.thresh_cog

        self.archive_limit = args.archive_limit
        self.archive_ilimit = args.archive_ilimit
        self.archive_gran = args.archive_gran
        self.archive_tour = args.archive_tour

        self.max_no_of_evals_so_far_fully_evaluated = args.max_evals
        self.genotypes_file = args.genotypes_file
        self.genotype_pool = None
        if self.genotypes_file is not None:
            self.genotype_pool = GenotypePool.from_file(
                self.genotypes_file,
                sample_replacement=args.genotypes_sample_replacement,
            )
        self.pop_size = args.pop_size #TODO we can split it later...
        self.max_pop_size = args.pop_size
        self.elite_pop_size = args.elite_pop_size
        self.swap_size = round(args.swap_perc*self.max_pop_size/100)

        self.no_of_evals_so_far = 0
        self.no_of_evals_so_far_within_constraints = 0
        self.no_of_evals_so_far_fully_evaluated = 0
        self.progress_log_interval = args.progress_log_interval
        self.generation = 0
        self.best_fit = None
        self.best_sol = None
        self.start_time = time.perf_counter()

        self.last_s_id = 0
        self.current_limit = 0
        self.constraint_max_len = 50
        self.max_substitution_ratio = args.max_substitution_ratio
        if self.max_substitution_ratio < 0:
            raise ValueError("--max_substitution_ratio must be non-negative")

        self.filepath = args.filename + self.create_filename(args)
        self.logger = Logger(self.filepath, self.verbose)

        self.features = [
            FeatureDeltaSpatial(),
            FeatureDeltaDirection(),
        ]

        self.feature_thresholds = [self.thresh_delta, self.thresh_cog]
        # self.features = [ FeatureParenthesis()]
        # self.feature_thresholds = [0]
        self.feature_extractor = FeatureExtractor(self.features, self.feature_thresholds, swap_intensity=self.swap_intensity, logger=self.logger)
        
        self.archive = CoSOArchive(limit=self.archive_limit, internal_limit=self.archive_ilimit, granularity=self.archive_gran, tournament=self.archive_tour)
        self.exp_logger = ExperimentLogger(self.logger, self._event_state)

        self.exp_logger.log_meta(args, self.features, self.feature_thresholds, self.filepath, task="vertpos")
        self.exp_logger.log_event("start")
    
    def increment_eval_counter(self):
        self.no_of_evals_so_far_fully_evaluated += 1

    def  parse_args(self):
        parser = argparse.ArgumentParser(description='conOGM')

        parser.add_argument('--frams_path', type=str, default="D:/Program Files (x86)/Framsticks",
                        help='Path to the Framsticks dir')
        parser.add_argument('--filename', type=str, default = "./",
                        help='Path to the file with results')
        parser.add_argument('--seed', type=int,
                        help='Seed of the run')
        parser.add_argument('--genotypes_file', type=str,
                        help='Path to a text file with one valid genotype per line')
        parser.add_argument('--genotypes_sample_replacement', action='store_true',
                        help='Sample genotypes from --genotypes_file with replacement')
            
        parser.add_argument('--max_evals', type=int, default=250_000,
                        help='Maximum number of evaluations')
        
        parser.add_argument('--pop_size', type=int, default=20,
                        help='Population size')
        parser.add_argument('--elite_pop_size', type=int, default=10,
                        help='Elite population size')
        parser.add_argument('--swap_perc', type=int, default=100,
                        help='Percent (as integer) of population swapped for new random solutions after convergence')
        parser.add_argument('--swap_intensity', type=int, default=0,
                        help='0 - no loop removal/insertion, 1 - loop removal/insertion, 2 - lr/i + single gene removal')
        parser.add_argument('--max_substitution_ratio', type=float, default=0.75,
                        help='Maximum replacement length as a fraction of the current genotype length')

        parser.add_argument('--thresh_delta', type=float, default=0,
                        help='Threshold for matching deltas (distance squared)')
        parser.add_argument('--thresh_cog', type=float, default=100,
                        help='Threshold for matching center of gravity (distance squared)')

        parser.add_argument('--archive_limit', type=int, default=5,
                        help='Maximum number of sequences returned from one bucket in the archive')
        parser.add_argument('--archive_ilimit', type=int, default=15,
                        help='Maximum number of sequences stored in one bucket in the archive')
        parser.add_argument('--archive_gran', type=float, default=10,
                        help='Granularity of the archive addressing (e.g. 3 means precision of 1/3 etc.)')
        parser.add_argument('--archive_tour', type=int, default=3,
                        help='Tournament size for selecting sequences from a bucket')
        parser.add_argument('--progress_log_interval', type=int, default=100,
                        help='Log progress every N full Framsticks evaluations; 0 disables interval progress logging')
        
        parser.add_argument('--verbose', type=int, default=1,
                        help='How much info should be written to standard output')
        return parser.parse_args()

    def create_filename(self, args):
        params = {
            "id": args.seed,
            "ps": args.pop_size,
            "eps": args.elite_pop_size,
            "sp": args.swap_perc,
            "si": args.swap_intensity,
            "msr": args.max_substitution_ratio,
            "dt": args.thresh_delta,
            "cogt": args.thresh_cog,
            "al": args.archive_limit,
            "ag": args.archive_gran,
            "ak": args.archive_tour,
            "ail": args.archive_ilimit,
            "gf": args.genotypes_file is not None,
            "gsr": args.genotypes_sample_replacement,
            "pli": args.progress_log_interval,
        }
        name = "results_" + str(params).replace(":", "~") + ".jsonl"
        return name


    def _elapsed_s(self):
        return time.perf_counter() - self.start_time


    def _archive_entry_count(self):
        return sum(len(entries) for entries in self.archive.archive.values())


    def _event_state(self):
        return {
            "generation": self.generation,
            "evals": self.no_of_evals_so_far,
            "within_constraint_evals": self.no_of_evals_so_far_within_constraints,
            "full_evals": self.no_of_evals_so_far_fully_evaluated,
            "elapsed_s": self._elapsed_s(),
            "best_fit": self.best_fit,
            "archive_keys": len(self.archive.archive),
            "archive_entries": self._archive_entry_count(),
        }


    def _update_best(self, fit, sol):
        if fit is None:
            return False

        if self.best_fit is None or fit > self.best_fit:
            self.best_fit = fit
            self.best_sol = sol
            return True

        return False


    def gen_starting_population(self):
        if self.genotype_pool is not None:
            return [self._register_new_solution(s) for s in self.genotype_pool.sample_many(self.pop_size)]
        
        return [self.get_random() for _ in range(self.pop_size)]

    def evaluate(self, sol):
        before_full_evals = self.no_of_evals_so_far_fully_evaluated
        self.no_of_evals_so_far += 1
        if len(sol) > self.constraint_max_len:
            fit = 0
            self._update_best(fit, sol)
            return fit

        if len(sol) == 0:
            fit = 0
            self._update_best(fit, sol)
            return fit

        self.no_of_evals_so_far_within_constraints += 1

        if self.genotype_pool is not None:
            known_fit = self.genotype_pool.get_fitness(sol)
            if known_fit is not None:
                self._update_best(known_fit, sol)
                return known_fit

        # return self.frams.evaluate("//9\n" + sol) #//9 necessary for f9
        fit = self.frams.evaluate(sol)
        is_new_best = self._update_best(fit, sol)
        self.exp_logger.maybe_log_eval_progress(
            before_full_evals,
            self.no_of_evals_so_far_fully_evaluated,
            self.progress_log_interval,
            fit,
        )
        if is_new_best:
            self.exp_logger.log_event("best", fit=fit, genotype=sol)
        return fit
    
    
    def _register_new_solution(self, s):
        self.last_s_id += 1
        s_id = self.last_s_id
        fit = self.evaluate(s)
        self.exp_logger.log_event("new", sol_id=s_id, fit=fit, genotype=s)

        return (s, s_id)


    def get_random(self):
        length = self.constraint_max_len
        if self.genotype_pool is not None:
            s = self.genotype_pool.sample()
        else:
            s = self.frams.random_solution(length)

        return self._register_new_solution(s)
    

    def update_archive(self, sol):
        dict = self.feature_extractor.extract(sol) #dict -> key: (feature1, feature2, ...), val: [(seq, (start, end)), ...]
        fit = self.evaluate(sol)
        for key in dict:
            self.archive.add(key, fit, dict[key].pop())

    def apply_sub(self, sol, sub):
        #TODO optimize
        return sol[:sub[0][0]] + sub[1] + sol[sub[0][1]:]


    def is_substitution_length_ok(self, genotype, replaced_fragment, replacement):
        max_replacement_len = int(len(genotype) * self.max_substitution_ratio)
        if len(replacement) > max_replacement_len:
            return False

        candidate_len = len(genotype) - len(replaced_fragment) + len(replacement)
        if candidate_len > self.constraint_max_len:
            return False

        return True


    def improve(self, solution1, sol_id, force = False):
        sol1_parts_dict = self.feature_extractor.extract(solution1, save_address=True)
        # print('DIIIIIIIIIIIIIIIIIII:', sol1_parts_dict)

        solution_fitness = self.evaluate(solution1)
        self.logger.print_verbose(1, "Improving", solution1, "(", solution_fitness, ")")   
        if solution_fitness is None:
            self.logger.print_verbose(1, "\tSkipping improvement due to None fitness")
            return solution1, False

        #dict1 -> key: (feature1, feature2, ...), val: [(seq, (start, end)), ...]
        keys = list(sol1_parts_dict.keys())
        random.shuffle(keys)
        archive_retrieve_stats = self.exp_logger.new_archive_retrieve_stats()
        for key in keys:
            vals = sol1_parts_dict[key]
            og_subs = self.archive.retrieve(
                key,
                fit= solution_fitness,
                thresholds = (self.thresh_delta, self.thresh_cog),
                limit=self.current_limit,
                force = force
            )
            self.exp_logger.record_archive_retrieve(archive_retrieve_stats, len(og_subs))
            self.exp_logger.log_archive_retrieve(
                sol_id,
                solution_fitness,
                force,
                [self.thresh_delta, self.thresh_cog],
                self.current_limit,
                key,
                len(og_subs),
                archive_retrieve_stats["calls"],
                len(vals),
            )
            for val in vals:
                address = val[1]
                subs = og_subs[:]
                
                if len(val[0]) <= 1:
                    subs.append("")
                    random.shuffle(subs)

                for s in subs:
                    if not self.is_substitution_length_ok(solution1, val[0], s):
                        continue

                    sub = (address, s)
                    candidate = self.apply_sub(solution1, sub)
                    unrepaired_candidate = candidate
                    candidate = self.frams.repair(candidate)
                    if candidate is None:
                        continue
                    repaired = candidate != unrepaired_candidate
                    candidate_fitness = self.evaluate(candidate)

                    if candidate_fitness is None:
                    #     print(candidate, ":(")
                        continue
                    # print(candidate, candidate_fitness)
                    
                    if candidate_fitness > solution_fitness: #TODO test >=
                        self.logger.print_verbose(
                            1,
                            "\tImprovement accepted!",
                            solution1, "(", solution_fitness, ") becomes",
                            candidate, "(", candidate_fitness, ")",
                            (solution1[sub[0][0]:sub[0][1]], sub[1]))
                        self.exp_logger.log_event(
                            "imp",
                            sol_id=sol_id,
                            fit=candidate_fitness,
                            previous_fit=solution_fitness,
                            genotype=candidate,
                            unrepaired_genotype=unrepaired_candidate if repaired else None,
                            repaired=repaired,
                            previous_genotype=solution1,
                            sub_from=solution1[sub[0][0]:sub[0][1]],
                            sub_to=sub[1],
                        )
                        self.exp_logger.log_progress("improvement", current_fit=candidate_fitness, current_sol_id=sol_id)
                        self.exp_logger.log_archive_retrieve_summary(
                            sol_id,
                            solution_fitness,
                            force,
                            [self.thresh_delta, self.thresh_cog],
                            self.current_limit,
                            archive_retrieve_stats,
                            accepted=True,
                            accepted_fit=candidate_fitness,
                        )
                        #update_archive(archive, candidate, candidate_fitness)
                        return candidate, True

        self.exp_logger.log_archive_retrieve_summary(
            sol_id,
            solution_fitness,
            force,
            [self.thresh_delta, self.thresh_cog],
            self.current_limit,
            archive_retrieve_stats,
            accepted=False,
        )
        return solution1, False #False means "no improvement"
        
    def evolve(self):

        current_limit = 1

        population = self.gen_starting_population()
        if self.genotype_pool is not None:
            pretender, pretender_id = population[0]
        else:
            pretender, pretender_id = self.get_random()

        elite_pop = []

        self.generation = 0

        FORCE_IMPROVEMENT = True
        IMPROVING_ELITES = False

        t = time.perf_counter_ns()

        while self.no_of_evals_so_far_fully_evaluated < self.max_no_of_evals_so_far_fully_evaluated: #for generation in range(1000000): # TODO while True, or directly check
            if FORCE_IMPROVEMENT:
                self.logger.print_verbose(1, "FORCING")
            #current_limit = 2+ int((archive_limit-1)*no_of_evals_so_far_fully_evaluated/max_no_of_evals_so_far_fully_evaluated)
            current_limit = self.archive_limit

            self.logger.print_verbose(1, "Generation #", self.generation, " limit ", str(current_limit))
           # print_verbose(1, "Best fitness (pop) = ", max([(self.evaluate(sol), sol) for sol, id in population], key=lambda x:x[0]))
            self.logger.print_verbose(1, "Best fitness (pretender) = ", (self.evaluate(pretender), pretender))
            self.logger.print_verbose(1, "Best fitness (elite_pop) = ", max([(self.evaluate(sol), sol) for sol, _ in elite_pop], key=lambda x:x[0], default=0))
            new_pop = []
            count = 0
            improvement = False
            
            if IMPROVING_ELITES:
                for sol, id in population:
                    self.logger.print_verbose(1, count, end=", ")
                    count += 1
                    new_sol, imp = self.improve(sol, id, force = True)
                    new_pop.append((new_sol, id))
                    improvement = improvement or imp
                population = new_pop
            else:
                self.logger.print_verbose(1, count, end=", ")
                pretender, imp = self.improve(pretender, pretender_id, force = FORCE_IMPROVEMENT)
                improvement = improvement or imp

            self.logger.print_verbose(1, pretender)
            self.logger.print_verbose(2, "no_of_evals_so_far", self.no_of_evals_so_far)
            self.logger.print_verbose(2, "no_of_evals_so_far_within_constraints", self.no_of_evals_so_far_within_constraints)
            self.logger.print_verbose(1, "no_of_evals_so_far_fully_evaluated", self.no_of_evals_so_far_fully_evaluated)

            if improvement and FORCE_IMPROVEMENT:
                FORCE_IMPROVEMENT = False

            if not improvement:
                if FORCE_IMPROVEMENT:
                    if self.evaluate(pretender) != None:
                        self.update_archive(pretender)

                    self.logger.print_verbose(1, 'el_pop:', elite_pop)
                   # print(pretender, self.evaluate(pretender))
                   #  if len(elite_pop):
                    bisect.insort_left(elite_pop, (pretender, pretender_id), key=lambda x: -self.evaluate(x[0])) #minus, because we want to sort from the highest fitness

                    while len(elite_pop) > self.elite_pop_size:
                        self.exp_logger.log_event(
                            "del",
                            sol_id=elite_pop[-1][1],
                            fit=self.evaluate(elite_pop[-1][0]),
                            genotype=elite_pop[-1][0],
                        )
                        elite_pop = elite_pop[:self.elite_pop_size]

                    self.exp_logger.log_event("converged", current_fit=self.evaluate(pretender), current_sol_id=pretender_id)
                    self.exp_logger.log_progress("converged", current_fit=self.evaluate(pretender), current_sol_id=pretender_id)
                    
                    pretender, pretender_id = self.get_random()
                    FORCE_IMPROVEMENT = False
                else:
                    FORCE_IMPROVEMENT = True

            self.exp_logger.log_progress("generation_end", current_fit=self.evaluate(pretender), current_sol_id=pretender_id)
            self.generation += 1

            progress_print_interval = max(1, self.max_no_of_evals_so_far_fully_evaluated // 10)
            if self.no_of_evals_so_far_fully_evaluated % progress_print_interval == 0:
                self.logger.print_verbose(0, "Kolejne 10%...", self.no_of_evals_so_far_fully_evaluated)

        elapsed_s = (time.perf_counter_ns() - t)/1e9
        print(elapsed_s)
        self.exp_logger.log_progress("finished", current_fit=self.evaluate(pretender), current_sol_id=pretender_id)
        self.exp_logger.log_event("fin", sol_id=pretender_id, fit=self.evaluate(pretender), genotype=pretender, rank=0)
        for sol, id in elite_pop:
            self.exp_logger.log_event("fin", sol_id=id, fit=self.evaluate(sol), genotype=sol)
        self.exp_logger.log_event("end", best_genotype=self.best_sol, total_elapsed_s=self._elapsed_s())

        self.logger.print_verbose(1, "Finally:")
        self.logger.print_verbose(0, "Best fitness = ", max([(self.evaluate(sol), (sol, id)) for sol, id in elite_pop], key=lambda x:x[0]))

if __name__ == '__main__':

    coso = CoSO()
    coso.evolve()
