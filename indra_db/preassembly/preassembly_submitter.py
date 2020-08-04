from indra.statements import get_all_descendants, Statement
from indra_reading.batch.submitters.submitter import Submitter


DEFAULT_AVOID_STATEMENTS = ['Event', 'Influence', 'Unresolved']
VALID_STATEMENTS = [st.__name__ for st in get_all_descendants(Statement)
                    if st.__name__ not in DEFAULT_AVOID_STATEMENTS]


class PreassemblySubmitter(Submitter):
    job_class = 'preassembly'
    _purpose = 'db_preassembly'
    _job_queue_dict = {'run_db_reading_queue': ['create', 'update']}
    _job_def_dict = {'run_db_reading_jobdef': ['create', 'update']}

    def __init__(self, basename, task, *args, **kwargs):
        if task not in ['create', 'update']:
            raise ValueError(f"Invalid task '{task}': expected 'create' or "
                             f"'update'.")
        self.task = task
        super(PreassemblySubmitter, self).__init__(basename, *args, **kwargs)

    def _iter_over_select_queues(self):
        for jq, tasks in self._job_queue_dict.items():
            if self.task not in tasks:
                continue
            yield jq

    def _get_command(self, job_type_set, stmt_type):
        if self.task not in job_type_set:
            return None, None
        job_name = f'{self.job_base}_{self.task}_{stmt_type}'
        cmd = ['python3', '-m', 'indra_db.preassembly.preassemble_db',
               self.task, '-n', '32', '-C', f'{self.s3_base}/{job_name}',
               '-T', stmt_type]
        return job_name, cmd

    def _iter_job_args(self, type_list=None):
        if type_list is None:
            type_list = VALID_STATEMENTS

        invalid_types = set(type_list) - set(VALID_STATEMENTS)
        if invalid_types:
            raise ValueError(f"Found invalid statement types: {invalid_types}")

        for stmt_type in type_list:
            yield (stmt_type,)

