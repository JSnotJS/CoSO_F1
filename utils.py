import json

class Logger:
    def __init__(self, filepath, verbose):
        self.filepath = filepath
        self.verbose = verbose

    def print_verbose(self, v, *args, **kwargs):
        if self.verbose >= v:
            print(*args, **kwargs)

    def _json_default(self, value):
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, set):
            return list(value)
        return str(value)

    def log_event(self, event, **fields):
        record = {"event": event}
        record.update(fields)

        with open(self.filepath, 'a', encoding='utf-8') as file:
            file.write(json.dumps(record, ensure_ascii=False, default=self._json_default) + '\n')

    def log_to_file(self, type, time=None, sol_id=None, sol=None, fit=None, sub_from=None, sub_to=None):
        fields = {}
        if time is not None:
            fields["time"] = time
        if sol_id is not None:
            fields["sol_id"] = sol_id
        if fit is not None:
            fields["fit"] = fit
        if sol is not None:
            fields["genotype"] = sol
        if sub_from is not None:
            fields["sub_from"] = sub_from
        if sub_to is not None:
            fields["sub_to"] = sub_to

        self.log_event(type, **fields)
