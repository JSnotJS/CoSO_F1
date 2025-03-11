
class Logger:
    def __init__(self, filepath, verbose):
        self.filepath = filepath
        self.verbose = verbose

    def print_verbose(self, v, *args, **kwargs):
        if self.verbose >= v:
            print(*args, **kwargs)

    def log_to_file(self, type, time = None, sol_id=None, sol=None, fit=None, sub_from=None, sub_to=None): 

        info_list = [type]
        if time != None:
            info_list.extend([str(time)])
        if sol_id != None and sol != None and fit != None:
            info_list.extend([str(sol_id), str(fit), sol])
        if sub_from != None and sub_to != None:
            info_list.extend([sub_from, sub_to])
        message = "; ".join(info_list)

        with open(self.filepath, 'a') as file:
            file.write(message+'\n')