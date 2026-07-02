from FramsticksLib import FramsticksLib
from functools import lru_cache 
import random

class FramsProblem:
    
    def __init__(self, frams_path, context_path, task, eval_increment_fun):
        self.frams = FramsticksLib(
            frams_path, 
            None, 
            ';'.join([
                f"{context_path}eval-allcriteria-mini.sim",
                f"{context_path}deterministic.sim",
                f"{context_path}sample-period-2.sim",
                f"{context_path}only-body.sim",
            ])
        )
        # "eval-allcriteria-mini.sim;deterministic.sim;sample-period-2.sim;only-body.sim" TODO

        if task == "vertpos" or task == "velocity":
            self.task = task
        else:
            print("Unknown task, defaulting to vertpos")
            self.task = "vertpos"

        self.increment_eval_counter = eval_increment_fun
    
    # def random_solution(self):
    #     return self.frams.getSimplest('9')
    
    def random_solution(self, length=20):
        s = self.frams.getSimplest('1')

        for _ in range(length):
            s = self.mutate(s)

        return s
    
    # lru_cache decorator allows us to memorize the output obtained from the last maxsize
    # (in our case 2048) calls of our method (with unique arguments).
    # as FramsProblem evaluation requires costly simulation, in practice lru_cache decorator can significantly 
    # speed up the evolution, as some of the tested solutions may have already been recently tested
    
    @lru_cache(maxsize=1_000_000, typed=False)
    def evaluate(self, s):
        # global no_of_evals_so_far_fully_evaluated
        # no_of_evals_so_far_fully_evaluated += 1

        self.increment_eval_counter() 

        fit = None
        try:
            result = self.frams.evaluate([s])[0]
            evaluations = result.get("evaluations") if isinstance(result, dict) else None
            if isinstance(evaluations, dict):
                default_eval = evaluations.get("")
                if isinstance(default_eval, dict):
                    fit = default_eval.get(self.task)
            if fit is None:
                print("[WARN] Missing evaluation data for genotype:", s)
        except Exception as e:
            print("[WARN] Evaluation exception for genotype:", s)
            # print("[WARN] Exception:", repr(e))
        return fit

    @lru_cache(maxsize=1_000_000, typed=False)
    def repair(self, s):
        return self.frams.repair(s)
    
    def mutate(self, s):
        return self.frams.mutate([s])[0]
    
    def crossover(self, s1, s2):
        return self.frams.crossOver(s1, s2)
