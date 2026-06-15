import bisect
import random
import numpy.random
import argparse
from CoSOArchive import CoSOArchive
from FramsProblem_VertposF1 import FramsProblem
from features.f1.FeatureExtractor import FeatureExtractor
from features.f1.FeatureDeltaSpatial import FeatureDeltaSpatial
from features.f1.FeatureDeltaDirection import FeatureDeltaDirection
from utils import Logger
import time

class CoSO:

    def __init__(self):
        args = self.parse_args()

        if args.seed == None:
            args.seed = random.randint(0, 100000)
        random.seed(args.seed)
        numpy.random.seed(args.seed)

        self.frams = FramsProblem(frams_path=args.frams_path, context_path="D:/STUDIA_v2/THEIR_MAGISTRY/CoSO/context/", task="vertpos", eval_increment_fun=self.increment_eval_counter)

        self.swap_intensity = args.swap_intensity
        self.verbose = args.verbose

        self.thresh_delta = args.thresh_delta
        self.thresh_cog = args.thresh_cog

        self.archive_limit = args.archive_limit
        self.archive_ilimit = args.archive_ilimit
        self.archive_gran = args.archive_gran
        self.archive_tour = args.archive_tour

        self.max_no_of_evals_so_far_fully_evaluated = args.max_evals
        self.population_zero = args.population_zero
        self.pop_size = args.pop_size #TODO we can split it later...
        self.max_pop_size = args.pop_size
        self.elite_pop_size = args.elite_pop_size
        self.swap_size = round(args.swap_perc*self.max_pop_size/100)

        self.no_of_evals_so_far = 0
        self.no_of_evals_so_far_within_constraints = 0
        self.no_of_evals_so_far_fully_evaluated = 0

        self.last_s_id = 0
        self.current_limit = 0
        self.constraint_max_len = 50

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

        self.logger.log_to_file("start")  
    
    def increment_eval_counter(self):
        self.no_of_evals_so_far_fully_evaluated += 1

    def parse_args(self):
        parser = argparse.ArgumentParser(description='conOGM')

        parser.add_argument('--frams_path', type=str, default="D:/Program Files (x86)/Framsticks",
                        help='Path to the Framsticks dir')
        parser.add_argument('--filename', type=str, default = "./",
                        help='Path to the file with results')
        parser.add_argument('--seed', type=int,
                        help='Seed of the run')
        parser.add_argument('--population_zero', type=int,
                        help='Path to the file with starting population candidates')
            
        parser.add_argument('--max_evals', type=int, default=250_000,
                        help='Maximum number of evaluations')
        
        parser.add_argument('--pop_size', type=int, default=20,
                        help='Population size')
        parser.add_argument('--elite_pop_size', type=int, default=10,
                        help='Elite population size')
        parser.add_argument('--swap_perc', type=int, default=100,
                        help='Percent (as integer) of population swapped for new random solutions after convergence')
        parser.add_argument('--swap_intensity', type=int, default=1,
                        help='0 - no loop removal/insertion, 1 - loop removal/insertion, 2 - lr/i + single gene removal')
        
        parser.add_argument('--thresh_delta', type=int, default=0,
                        help='Threshold for matching deltas (distance squared)')
        parser.add_argument('--thresh_cog', type=int, default=100,
                        help='Threshold for matching center of gravity (distance squared)')
        

        parser.add_argument('--archive_limit', type=int, default=5,
                        help='Maximum number of sequences returned from one bucket in the archive')
        parser.add_argument('--archive_ilimit', type=int, default=15,
                        help='Maximum number of sequences stored in one bucket in the archive')
        parser.add_argument('--archive_gran', type=float, default=1,
                        help='Granularity of the archive addressing (e.g. 3 means precision of 1/3 etc.)')
        parser.add_argument('--archive_tour', type=int, default=3,
                        help='Tournament size for selecting sequences from a bucket')
        
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
            "dt": args.thresh_delta,
            "cogt": args.thresh_cog,
            "al": args.archive_limit,
            "ag": args.archive_gran,
            "ak": args.archive_tour,
            "ail": args.archive_ilimit
        }

        name = "results_" + str(params).replace(":", "~") + ".txt" 

        return name


    def gen_starting_population(self):
        if self.population_zero is not None:
            ... # TODO wyciąganie z pliku population_zero
        
        return [self.get_random() for _ in range(self.pop_size)]

    def evaluate(self, sol):

        self.no_of_evals_so_far += 1
        if len(sol) > self.constraint_max_len:
            return 0

        if len(sol) == 0:
            return 0

        self.no_of_evals_so_far_within_constraints += 1

        # return self.frams.evaluate("//9\n" + sol) #//9 necessary for f9
        return self.frams.evaluate(sol)
    
    
    def get_random(self):
        length = self.constraint_max_len
        s = self.frams.random_solution(length)

        self.last_s_id += 1
        s_id = self.last_s_id
        self.logger.log_to_file(type="new", sol=s, sol_id=s_id, fit=self.evaluate(s))

        return (s, s_id)
    

    def update_archive(self, sol):
        dict = self.feature_extractor.extract(sol) #dict -> key: (deltas, cog), [val: seq]
        fit = self.evaluate(sol)
        for key in dict:
            self.archive.add(key, fit, dict[key].pop())

    def apply_sub(self, sol, sub):
        #TODO optimize
        return sol[:sub[0][0]] + sub[1] + sol[sub[0][1]:]

    def improve(self, solution1, sol_id, force = False):

        dict = self.feature_extractor.extract(solution1, save_address=True)
        # print('DIIIIIIIIIIIIIIIIIII:', dict)

        solution_fitness = self.evaluate(solution1)
        self.logger.print_verbose(1, "Improving", solution1, "(", solution_fitness, ")")   
        if solution_fitness is None:
            self.logger.print_verbose(1, "\tSkipping improvement due to None fitness")
            return solution1, False

        #dict1 -> key: (spacial, direction), [val: (seq, address)]
        keys = list(dict.keys())
        random.shuffle(keys)
        for key in keys:
            vals = dict[key]
            og_subs = self.archive.retrieve(key, fit= solution_fitness, thresholds = (self.thresh_delta, self.thresh_cog), limit=self.current_limit, force = force)
            for val in vals:
                address = val[1]
                subs = og_subs[:]
                
                if len(val[0]) <= 1:
                    subs.append("")
                    random.shuffle(subs)

                for s in subs:
                    sub = (address, s)
                    candidate = self.apply_sub(solution1, sub)
                    candidate_fitness = self.evaluate(candidate)

                    if candidate_fitness is None:
                    #     print(candidate, ":(")
                        continue
                    # print(candidate, candidate_fitness)
                    
                    if candidate_fitness > solution_fitness: #TODO test >=
                        self.logger.print_verbose(1, "\tImprovement accepted!", solution1, "(", solution_fitness, ") becomes", candidate, "(", candidate_fitness, ")", (solution1[sub[0][0]:sub[0][1]], sub[1]))
                        self.logger.log_to_file(type="imp", sol=candidate, sol_id=sol_id, fit=candidate_fitness, sub_from=solution1[sub[0][0]:sub[0][1]], sub_to=sub[1])
                        #update_archive(archive, candidate, candidate_fitness)
                        return candidate, True

        return solution1, False #False means "no improvement"        
        
    def evolve(self):

        current_limit = 1

        population = self.gen_starting_population()
        pretender, pretender_id = self.get_random()

        elite_pop = []

        generation = 0

        FORCE_IMPROVEMENT = True
        IMPROVING_ELITES = False

        t = time.perf_counter_ns()

        while self.no_of_evals_so_far_fully_evaluated < self.max_no_of_evals_so_far_fully_evaluated: #for generation in range(1000000): # TODO while True, or directly check
            if FORCE_IMPROVEMENT:
                self.logger.print_verbose(1, "FORCING")
            #current_limit = 2+ int((archive_limit-1)*no_of_evals_so_far_fully_evaluated/max_no_of_evals_so_far_fully_evaluated)
            current_limit = self.archive_limit

            self.logger.print_verbose(1, "Generation #", generation, " limit ", str(current_limit))
            generation +=1
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
                        self.logger.log_to_file(type = "del", sol = elite_pop[-1][0], sol_id = elite_pop[-1][1], fit = self.evaluate(elite_pop[-1][0]))
                        elite_pop = elite_pop[:self.elite_pop_size]
                    
                    self.logger.log_to_file("converged")
                    
                    pretender, pretender_id = self.get_random()
                    FORCE_IMPROVEMENT = False
                else:
                    FORCE_IMPROVEMENT = True

        print((time.perf_counter_ns() - t)/1e9)
        self.logger.log_to_file(type = "fin", sol = pretender, sol_id = pretender_id, fit = self.evaluate(pretender))
        for sol, id in elite_pop:
            self.logger.log_to_file(type = "fin", sol = sol, sol_id = id, fit = self.evaluate(sol))
        self.logger.log_to_file("end")

        self.logger.print_verbose(1, "Finally:")
        self.logger.print_verbose(0, "Best fitness = ", max([(self.evaluate(sol), (sol, id)) for sol, id in elite_pop], key=lambda x:x[0]))

if __name__ == '__main__':

    coso = CoSO()
    coso.evolve()
