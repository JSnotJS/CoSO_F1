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


class ExperimentLogger:
    def __init__(self, logger, state_provider):
        self.logger = logger
        self.state_provider = state_provider
        self.last_progress_full_evals = -1

    def log_event(self, event, **fields):
        payload = self.state_provider()
        payload.update(fields)
        self.logger.log_event(event, **payload)

    def log_meta(self, args, features, thresholds, result_file, task):
        self.logger.log_event(
            "meta",
            schema_version=1,
            seed=args.seed,
            task=task,
            frams_path=args.frams_path,
            max_evals=args.max_evals,
            pop_size=args.pop_size,
            elite_pop_size=args.elite_pop_size,
            swap_perc=args.swap_perc,
            swap_intensity=args.swap_intensity,
            max_substitution_ratio=args.max_substitution_ratio,
            thresholds=thresholds,
            archive_limit=args.archive_limit,
            archive_internal_limit=args.archive_ilimit,
            archive_granularity=args.archive_gran,
            archive_tournament=args.archive_tour,
            progress_log_interval=args.progress_log_interval,
            genotypes_file=args.genotypes_file,
            genotypes_sample_replacement=args.genotypes_sample_replacement,
            features=[feature.__class__.__name__ for feature in features],
            result_file=result_file,
        )

    def log_progress(self, reason, current_fit=None, current_sol_id=None):
        self.log_event(
            "progress",
            reason=reason,
            current_fit=current_fit,
            current_sol_id=current_sol_id,
        )

    def maybe_log_eval_progress(self, before_full_evals, current_full_evals, interval, fit):
        if interval <= 0:
            return

        if current_full_evals == before_full_evals:
            return

        if current_full_evals == self.last_progress_full_evals:
            return

        if current_full_evals % interval == 0:
            self.last_progress_full_evals = current_full_evals
            self.log_progress("eval_interval", current_fit=fit)

    def new_archive_retrieve_stats(self):
        return {
            "calls": 0,
            "candidates_total": 0,
            "nonempty_calls": 0,
            "max_candidates": 0,
        }

    def record_archive_retrieve(self, stats, candidates_count):
        stats["calls"] += 1
        stats["candidates_total"] += candidates_count
        if candidates_count > 0:
            stats["nonempty_calls"] += 1
        if candidates_count > stats["max_candidates"]:
            stats["max_candidates"] = candidates_count

    def log_archive_retrieve_summary(self, sol_id, solution_fitness, force, thresholds, retrieve_limit, stats, accepted, accepted_fit=None):
        calls = stats["calls"]
        mean_candidates = stats["candidates_total"] / calls if calls > 0 else 0
        self.log_event(
            "archive_retrieve_summary",
            sol_id=sol_id,
            fit=solution_fitness,
            force=force,
            thresholds=thresholds,
            retrieve_limit=retrieve_limit,
            archive_retrieve_calls=calls,
            archive_retrieve_candidates_total=stats["candidates_total"],
            archive_retrieve_nonempty_calls=stats["nonempty_calls"],
            archive_retrieve_max_candidates=stats["max_candidates"],
            archive_retrieve_candidates_mean=mean_candidates,
            accepted=accepted,
            accepted_fit=accepted_fit,
        )

    def log_archive_retrieve(self, sol_id, solution_fitness, force, thresholds, retrieve_limit, key, candidates_count, retrieve_index, matching_targets_count):
        self.log_event(
            "archive_retrieve",
            sol_id=sol_id,
            fit=solution_fitness,
            force=force,
            thresholds=thresholds,
            retrieve_limit=retrieve_limit,
            archive_retrieve_index=retrieve_index,
            archive_retrieve_candidates=candidates_count,
            archive_retrieve_matching_targets=matching_targets_count,
            feature_key=key,
        )
