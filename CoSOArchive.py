import bisect
import random

class CoSOArchive:
    
    def __init__(self, limit = 5, internal_limit = 15, granularity = 1, tournament = 3):
        self.archive = {}
        self.limit = limit
        self.internal_limit = internal_limit
        self.granularity = granularity

        self.k = tournament

    def sanitize_key(self, key):
        new_key = []
        for p in key:
            els = []
            for el in p:
                els.append(float(int(float(el)*self.granularity)/self.granularity))
            new_key.append(tuple(els))
        return tuple(new_key)

    def add(self, key, fit, val):
        #print(self.archive)

        key = self.sanitize_key(key)

        entry = (fit, val)

        if key not in self.archive:
            self.archive[key] = []

        if val in [v[1] for v in self.archive[key]]:
            #TODO recalculate fitness
            for v in self.archive[key]:
                if v[1] == val and fit > v[0]:
                    self.archive[key].remove(v) #we will reinsert it at a new position in a sec

        bisect.insort_left(self.archive[key], entry, key=lambda x: -x[0]) #minus, because we want to sort from the highest fitness

        self.archive[key] = self.archive[key][:self.internal_limit] #remove the worst performing entries
        #self.archive[key] = self.archive[key][:self.limit] #remove the worst performing entries
        

    def retrieve(self, key, fit = None, limit=None, thresholds=None, force=False):
        
        key = self.sanitize_key(key)

        def only_vals(l, f):
           # print(l)
            vls = [v[1] for v in l]# if v[0] > 0.5 * f])
            #vls = [(v[1] for v in l)]# if v[0] > 0.5 * f])
            return vls

        def only_vals_tour(l, lim):
            vls = []            
            cp_l = l[:]
            for _ in range(lim):
                if len(cp_l) <= 0:
                    break
                v = max([cp_l[random.randint(0, len(cp_l)-1)] for i in range(self.k)], key=lambda x:x[0])
                cp_l.remove(v)
                vls.append(v[1])
            return vls
            
        def feature_distance(vec1, vec2):
            if len(vec1) != len(vec2):
                return float("inf")
            return sum((float(vec1[i]) - float(vec2[i])) ** 2 for i in range(len(vec1))) ** 0.5
        
        def matches(key1, key2):
            if len(key1) == len(key2):
                for i in range(len(key1)):
                    if feature_distance(key1[i], key2[i]) > thresholds[i]:
                        return False
                return True
            return False

        if thresholds == None:
            thresholds = [0 for _ in range(len(key))]
        if len(thresholds) != len(key):
            return [] #TODO do it differently perhaps?

        if limit == None:
            limit = self.limit
        
        subs = set()
        for fkey in self.archive:
            if matches(key, fkey) and fkey in self.archive:
                if force:
                    vals = only_vals(self.archive[fkey], fit) #return all
                else:
                    vals = only_vals_tour(self.archive[fkey], limit) #return some, selected with tournament
                subs.update(vals)

        subs = list(subs)
        return subs
