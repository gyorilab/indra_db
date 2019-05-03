""""This file acts as a script to run large batch jobs on AWS.

The key components are the DbReadingSubmitter class, and the submit_db_reading
function. The function is provided as a shallow wrapper for backwards
compatibility, and may eventually be removed. The preferred method for running
large batches via the ipython, or from a python environment, is the following:

>> sub = DbReadingSubmitter('name_for_run', ['reach', 'sparser'])
>> sub.set_options(prioritize=True)
>> sub.submit_reading('file/location/of/ids_to_read.txt', 0, None, ids_per_job=1000)
>> sub.watch_and_wait(idle_log_timeout=100, kill_on_timeout=True)

Additionally, this file may be run as a script. For details, run

bash$ python submit_reading_pipeline.py --help

In your favorite command line.
"""
from collections import defaultdict

import re
import boto3
import pickle
import logging
from numpy import median, arange, array, zeros
import matplotlib as mpl; mpl.use('Agg')
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

from indra.util.get_version import get_git_info
from indra.util.nested_dict import NestedDict
from indra.util.aws import get_s3_file_tree
from indra.tools.reading.util.reporter import Reporter
from indra.tools.reading.submit_reading_pipeline import create_submit_parser, \
    create_read_parser, Submitter

bucket_name = 'bigmech'

logger = logging.getLogger('indra_db_reading')


class DbReadingSubmitter(Submitter):
    """A class for the management of a batch of reading jobs on AWS.

    Parameters
    ----------
    basename : str
        The name of this batch of readings. This will be used to distinguish
        the jobs on AWS batch, and the logs on S3.
    readers : list[str]
        A list of the names of readers to use in this job.
    project_name : str
        Optional. Used for record-keeping on AWS.

    Other keyword parameters go to the `get_options` method.
    """
    _s3_input_name = 'id_list'
    _purpose = 'db_reading'
    _job_queue = 'run_db_reading_queue'
    _job_def = 'run_db_reading_jobdef'

    def __init__(self, *args, **kwargs):
        super(DbReadingSubmitter, self).__init__(*args, **kwargs)
        self.time_tag = datetime.now().strftime('%Y%m%d_%H%M')
        self.reporter = Reporter(self.basename + '_summary_%s' % self.time_tag)
        self.reporter.sections = {'Plots': [], 'Totals': [], 'Git': []}
        self.reporter.set_section_order(['Git', 'Totals', 'Plots'])
        self.run_record = {}
        self.start_time = None
        self.end_time = None
        return

    def submit_reading(self, *args, **kwargs):
        self.start_time = datetime.utcnow()
        super(DbReadingSubmitter, self).submit_reading(*args, **kwargs)
        return

    def _get_base(self, job_name, start_ix, end_ix):
        read_mode = self.options.get('reading_mode', 'unread')
        stmt_mode = self.options.get('stmt_mode', 'all')

        job_name = '%s_%d_%d' % (self.basename, start_ix, end_ix)
        base = ['python', '-m', 'indra_db.reading.read_db_aws',
                self.basename]
        base += [job_name]
        base += ['/tmp', read_mode, stmt_mode, '32', str(start_ix), str(end_ix)]
        return base

    def _get_extensions(self):
        extensions = []
        for key, val in self.options.items():
            if val is not None:
                extensions.append(['--' + key, val])
        return extensions

    def set_options(self, stmt_mode='all', reading_mode='unread',
                    max_reach_input_len=None, max_reach_space_ratio=None):
        """Set the options for this reading job.

        Parameters
        ----------
        stmt_mode : bool
            Optional, default 'all' - If 'all', produce statements for all
            content for all readers. If the readings were already produced,
            they will be retrieved from the database if `read_mode` is 'none'
            or 'unread'. If this option is 'unread', only the newly produced
            readings will be processed. If 'none', no statements will be
            produced.
        reading_mode : str : 'all', 'unread', or 'none'
            Optional, default 'undread' - If 'all', read everything (generally
            slow); if 'unread', only read things that were unread, (the cache
            of old readings may still be used if `stmt_mode='all'` to get
            everything); if 'none', don't read, and only retrieve existing
            readings.
        max_reach_input_len : int
            The maximum number of characters to all for inputs to REACH. The
            reader tends to hang up on larger papers, and beyond a certain
            threshold, greater length tends to imply errors in formatting or
            other quirks.
        max_reach_space_ratio : float in [0,1]
            Some content erroneously has spaces between all characters. The
            fraction of characters that are spaces is a fast and simple way to
            catch and avoid such problems. Recommend a value of 0.5.
        """
        self.options['stmt_mode'] = stmt_mode
        self.options['reading_mode'] = reading_mode
        self.options['max_reach_input_len'] = max_reach_input_len
        self.options['max_reach_space_ratio'] = max_reach_space_ratio
        return

    def watch_and_wait(self, *args, **kwargs):
        """Watch the logs of the batch jobs and wait for all jobs to complete.

        Logs are monitored, and jobs may be killed if no output is seen for a
        given amount of time. Essential if jobs are left un-monitored (by
        humans) for any length of time.

        Parameters
        ----------
        poll_interval: int
            Default 10. The number of seconds to wait between examining logs and
            the states of the jobs.
        idle_log_timeout : int or None,
            Default is None. If an int, sets the number of seconds to wait
            before declaring a job timed out. This parameter alone does not lead
            to the deletion/stopping of stalled jobs.
        kill_on_timeout : bool
            Default is False. If true, and a job is deemed to have timed out,
            kill the job.
        stash_log_method : str or None
            Default is None. If a string is given, logs will be saved in the
            indicated location. Value choices are 's3' or 'local'.
        tag_instances : bool
            Default is False. In the past, it was necessary to tag instances
            from the outside. THey should now be tagging themselves, however if
            for any reason external measures are needed, this option may be set
            to True.
        """
        kwargs['result_record'] = self.run_record
        super(DbReadingSubmitter, self).watch_and_wait(*args, **kwargs)
        self.end_time = datetime.utcnow()
        if self.job_list:
            self.produce_report()

    @staticmethod
    def _parse_time(time_str):
        """Create a timedelta or datetime object from default string reprs."""
        try:
            # This is kinda terrible, but it is the easiest way to distinguish
            # them.
            if '-' in time_str:
                time_fmt = '%Y-%m-%d %H:%M:%S'
                if '.' in time_str:
                    pre_dec, post_dec = time_str.split('.')
                    dt = datetime.strptime(pre_dec, time_fmt)
                    dt.replace(microsecond=int(post_dec))
                else:
                    dt = datetime.strftime(time_str, time_fmt)
                return dt
            else:
                if 'day' in time_str:
                    m = re.match(('(?P<days>[-\d]+) day[s]*, '
                                  '(?P<hours>\d+):(?P<minutes>\d+):'
                                  '(?P<seconds>\d[\.\d+]*)'),
                                 time_str)
                else:
                    m = re.match(('(?P<hours>\d+):(?P<minutes>\d+):'
                                  '(?P<seconds>\d[\.\d+]*)'),
                                 time_str)
                return timedelta(**{key: float(val)
                                    for key, val in m.groupdict().items()})
        except Exception as e:
            logger.error('Failed to parse \"%s\".' % time_str)
            raise e

    def _get_txt_file_dict(self, file_bytes):
        line_list = file_bytes.decode('utf-8').splitlines()
        sc = ': '
        file_info = {}
        for line in line_list:
            segments = line.split(sc)
            file_info[segments[0].strip()] = sc.join(segments[1:]).strip()
        return file_info

    def _handle_git_info(self, ref, git_info, file_bytes):
        this_info = self._get_txt_file_dict(file_bytes)
        if git_info and this_info != git_info:
            logger.warning("Disagreement in git info in %s: "
                           "%s vs. %s."
                           % (ref, git_info, this_info))
        elif not git_info:
            git_info.update(this_info)
        return

    def _report_git_info(self, batch_git_info):
        self.reporter.add_text('Batch Git Info', section='Git', style='h1')
        for key, val in batch_git_info.items():
            label = key.replace('_', ' ').capitalize()
            self.reporter.add_text('%s: %s' % (label, val), section='Git')
        self.reporter.add_text('Launching System\'s Git Info', section='Git',
                               style='h1')
        git_info_dict = get_git_info()
        for key, val in git_info_dict.items():
            label = key.replace('_', ' ').capitalize()
            self.reporter.add_text('%s: %s' % (label, val), section='Git')
        return

    def _handle_timing(self, ref, timing_info, file_bytes):
        this_info = self._get_txt_file_dict(file_bytes)
        for stage, data in this_info.items():
            if stage not in timing_info.keys():
                logger.info("Adding timing stage: %s" % stage)
                timing_info[stage] = {}
            stage_info = timing_info[stage]
            timing_pairs = re.findall(r'(\w+):\s+([ 0-9:.\-]+)', data)
            if len(timing_pairs) is not 3:
                logger.warning("Not all timings present for %s "
                               "in %s." % (stage, ref))
            for label, time_str in timing_pairs:
                if label not in stage_info.keys():
                    stage_info[label] = {}
                # e.g. timing_info['reading']['start']['job_name'] = <datetime>
                stage_info[label][ref] = self._parse_time(time_str)
        return

    def _report_timing(self, timing_info):
        # Pivot the timing info.
        idx_patt = re.compile('%s_(\d+)_(\d+)' % self.basename)
        plot_set = set()

        def add_job_to_plot_set(job_name):
            m = idx_patt.match(job_name)
            if m is None:
                logger.error("Unexpectedly formatted name: %s."
                             % job_name)
                return None
            key = tuple([int(n) for n in m.groups()] + [job_name])
            plot_set.add(key)

        job_segs = NestedDict()
        stages = []
        for stage, stage_d in timing_info.items():
            # e.g. reading, statement production...
            stages.append((list(stage_d['start'].values())[0], stage))
            for metric, metric_d in stage_d.items():
                # e.g. start, end, ...
                for job_name, t in metric_d.items():
                    # e.g. job_basename_startIx_endIx
                    job_segs[job_name][stage][metric] = t
                    add_job_to_plot_set(job_name)
        stages = [stg for _, stg in sorted(stages)]

        # Add data from local records, if available.
        if self.run_record:
            for final_status in ['failed', 'succeeded']:
                for job in self.run_record[final_status]:
                    if job['jobName'] in job_segs.keys():
                        job_d = job_segs[job['jobName']]
                    else:
                        job_d = {}
                        job_segs[job['jobName']] = job_d
                        add_job_to_plot_set(job['jobName'])
                    terminated = job['jobId'] in self.run_record['terminated']
                    job_d['final'] = final_status
                    job_d['terminated'] = terminated
                    for time_key in ['createdAt', 'startedAt', 'stoppedAt']:
                        stage = 'job_' + time_key.replace('At', '')
                        ts = job.get(time_key)
                        if ts is None:
                            job_d[stage] = ts
                        else:
                            job_d[stage] = \
                                datetime.utcfromtimestamp(job[time_key]/1000)

        # Handle the start and end times.
        if self.start_time is None or self.end_time is None:
            all_times = [dt for job in job_segs.values() for stage in job.values()
                         for metric, dt in stage.items() if metric != 'duration']

            if self.start_time is None:
                self.start_time = min(all_times)

            if self.end_time is None:
                self.end_time = max(all_times)

        # Use this for getting the minimum and maximum.
        t_gap = 5

        def get_time_tuple(stage_data):
            start_seconds = (stage_data['start'] - self.start_time).total_seconds()
            dur = stage_data['duration'].total_seconds()
            if start_seconds > t_gap:
                start_seconds += t_gap/2

            if dur > t_gap:
                dur -= t_gap
            else:
                dur = t_gap

            return start_seconds, dur

        def make_y(start, end, scale):
            h = end - start
            start += (1 - scale)*h/2
            return start, h*scale

        label_size = 5
        # Make the broken barh plots from internal info -----------------------
        w = 6.5
        h = 9
        fig = plt.figure(figsize=(w, h), dpi=600)
        gs = plt.GridSpec(2, 1, height_ratios=[10, 0.7])
        ax0 = plt.subplot(gs[0])
        ytick_pairs = []
        total_time = (self.end_time - self.start_time).total_seconds()
        t = arange(total_time)

        # Initialize counts
        counts = defaultdict(lambda: {'data': zeros(len(t)), 'color': None})

        # Plot data from remote resources
        for i, job_tpl in enumerate(sorted(plot_set)):
            s_ix, e_ix, job_name = job_tpl
            job_d = job_segs[job_name]

            # Plot the overall job run info, if known.
            if self.run_record:
                ts = [None if job_d[k] is None
                      else (job_d[k] - self.start_time).total_seconds()
                      for k in ['job_created', 'job_started', 'job_stopped']]
                if ts[1] is None:
                    xs = [(ts[0], ts[2] - ts[0])]
                    facecolors = ['lightgray']
                elif None not in ts:
                    xs = [(ts[0], ts[1] - ts[0]), (ts[1], ts[2] - ts[1])]
                    facecolors = ['lightgray', 'gray']
                else:
                    xs = []
                    facecolors = []
                    print("Unhandled.")
                ys = make_y(s_ix, e_ix, 0.9)
                ax0.broken_barh(xs, ys, facecolors=facecolors)

            ytick_pairs.append(((s_ix + e_ix)/2, '%s_%s' % (s_ix, e_ix),
                                job_name))

            if job_d['final'] == 'failed':
                continue

            # Plot the more detailed info
            xs = [get_time_tuple(job_d.get(stg)) for stg in stages]
            ys = make_y(s_ix, e_ix, 0.4)
            logger.debug("Making plot for: %s" % str((job_name, xs, ys)))
            facecolors = [get_stage_choices(stg)[1] for stg in stages]
            ax0.broken_barh(xs, ys, facecolors=facecolors)

            for n, stg in enumerate(stages + ['jobs']):
                label, color = get_stage_choices(stg)
                counts[label]['color'] = color
                cs = counts[label]['data']
                if stg != 'jobs':
                    start = xs[n][0]
                    dur = xs[n][1]
                    cs[(t > start) & (t < (start + dur))] += 1
                else:
                    cs[(t > xs[0][0]) & (t < (xs[-1][0] + xs[-1][1]))] += 1

        # Format the plot
        ax0.tick_params(top='off', left='off', right='off', bottom='off',
                        labelleft='on', labelbottom='off')
        for spine in ax0.spines.values():
            spine.set_visible(False)
        ax0.set_xlim(0, total_time)
        ax0.set_ylabel(self.basename + '_ ...')
        yticks, ylabels, names = zip(*ytick_pairs)
        if not self.ids_per_job:
            print([yticks[i+1] - yticks[i]
                   for i in range(len(yticks) - 1)])
            # Infer if we don't have it.
            spacing = median([yticks[i+1] - yticks[i]
                              for i in range(len(yticks) - 1)])
            spacing = max([1, spacing])
        else:
            spacing = self.ids_per_job

        ytick_range = list(arange(yticks[0], yticks[-1] + spacing, spacing))
        ylabel_filled = []
        colored_labels = {}
        for i, ytick in enumerate(ytick_range):
            if ytick in yticks:
                ylabel = ylabels[yticks.index(ytick)]
                job_d = job_segs[names[yticks.index(ytick)]]
                if job_d.get('terminated'):
                    ylabel = 'T ' + ylabel
                    colored_labels[i] = 'red'
                elif job_d.get('final') == 'failed':
                    ylabel = 'F ' + ylabel
                    colored_labels[i] = 'orange'
                ylabel_filled.append(ylabel)
            else:
                ylabel_filled.append('FAILED')
        ax0.set_ylim(0, max(ytick_range) + spacing)
        ax0.set_yticks(ytick_range)
        ax0.set_yticklabels(ylabel_filled)
        ax0.tick_params(labelsize=label_size)
        for i, ytick in enumerate(ax0.get_yticklabels()):
            if i in colored_labels.keys():
                ytick.set_color(colored_labels[i])

        # Plot the lower axis -------------------------------------------------
        ax1 = plt.subplot(gs[1], sharex=ax0)

        # make the plot
        for label, cd in counts.items():
            ax1.plot(t, cd['data'], color=cd['color'], label=label)

        # Remove the axis bars.
        for lbl, spine in ax1.spines.items():
            spine.set_visible(False)

        # Format the plot.
        max_n = int(counts['total']['data'].max())
        ax1.set_ylim(0, max_n + 1)
        ax1.set_xlim(0, total_time)
        yticks = list(range(0, max_n-max_n//5, max(1, max_n//5)))
        ax1.set_yticks(yticks + [max_n])
        ax1.set_yticklabels([str(n) for n in yticks] + ['max=%d' % max_n])
        ax1.set_ylabel('N_jobs')
        ax1.set_xlabel('Time since beginning [seconds]')
        ax1.tick_params(labelsize=label_size)
        ax1.legend(loc='best', fontsize=label_size)

        # Make the figure borders more sensible -------------------------------
        fig.tight_layout()
        img_path = 'time_figure.png'
        fig.savefig(img_path)
        self.reporter.add_image(img_path, width=w, height=h, section='Plots')
        return dict(counts)

    def _handle_sum_data(self, job_ref, summary_info, file_bytes):
        one_sum_data_dict = pickle.loads(file_bytes)
        for k, v in one_sum_data_dict.items():
            if k not in summary_info.keys():
                summary_info[k] = {}
            summary_info[k][job_ref] = v
        return

    def _report_sum_data(self, summary_info):
        # Two kind of things to handle:
        for k, job_dict in summary_info.items():
            if isinstance(list(job_dict.values())[0], dict):
                continue

            # Overall totals
            self.reporter.add_text('total %s: %d' % (k, sum(job_dict.values())),
                                   section='Totals')

            # Hists of totals.
            if len(job_dict) <= 1:
                continue

            w = 6.5
            h = 4
            fig = plt.figure(figsize=(w, h))
            plt.hist(list(job_dict.values()), align='left')
            plt.xlabel(k)
            plt.ylabel('Number of Jobs')
            fig.tight_layout()
            fname = k + '_hist.png'
            fig.savefig(fname)
            self.reporter.add_image(fname, width=w, height=h, section='Plots')
        return

    def _handle_hist_data(self, job_ref, hist_dict, file_bytes):
        a_hist_data_dict = pickle.loads(file_bytes)
        for k, v in a_hist_data_dict.items():
            if k not in hist_dict.keys():
                hist_dict[k] = {}
            hist_dict[k][job_ref] = v
        return

    def _report_hist_data(self, hist_dict):
        for k, data_dict in hist_dict.items():
            w = 6.5
            if k == ('stmts', 'readers'):
                h = 6
                fig = plt.figure(figsize=(w, h))
                data = {}
                for job_datum in data_dict.values():
                    for rdr, num in job_datum['data'].items():
                        if rdr not in data.keys():
                            data[rdr] = [num]
                        else:
                            data[rdr].append(num)
                N = len(data)
                key_list = list(data.keys())
                xtick_locs = arange(N)
                n = (N+1)*100 + 11
                ax0 = plt.subplot(n)
                ax0.bar(xtick_locs, [sum(data[k]) for k in key_list],
                        align='center')
                ax0.set_xticks(xtick_locs, key_list)
                ax0.set_xlabel('readers')
                ax0.set_ylabel('stmts')
                ax0.set_title('Reader production')
                rdr_ax_list = []
                for rdr, stmt_counts in data.items():
                    n += 1
                    if not rdr_ax_list:
                        ax = plt.subplot(n)
                    else:
                        ax = plt.subplot(n, sharex=rdr_ax_list[0])
                    ax.set_title(rdr)
                    ax.hist(stmt_counts, align='left')
                    ax.set_ylabel('jobs')
                    rdr_ax_list.append(ax)
                if rdr_ax_list:
                    ax.set_xlabel('stmts')
            else:  # TODO: Handle other summary plots.
                continue
            figname = '_'.join(k) + '.png'
            fig.savefig(figname)
            self.reporter.add_image(figname, width=w, height=h, section='Plots')

        return

    def produce_report(self):
        """Produce a report of the batch jobs."""
        s3_prefix = 'reading_results/%s/logs/%s/' % (self.basename,
                                                     self._job_queue)
        logger.info("Producing batch report for %s, from prefix %s."
                    % (self.basename, s3_prefix))
        s3 = boto3.client('s3')
        file_tree = get_s3_file_tree(s3, bucket_name, s3_prefix)
        logger.info("Found %d relevant files." % len(file_tree))
        stat_files = {
            'git_info.txt': (self._handle_git_info, self._report_git_info),
            'timing.txt': (self._handle_timing, self._report_timing),
            'raw_tuples.pkl': (None, None),
            'hist_data.pkl': (self._handle_hist_data, self._report_hist_data),
            'sum_data.pkl': (self._handle_sum_data, self._report_sum_data)
            }
        stat_aggs = {}
        for stat_file, (handle_stats, report_stats) in stat_files.items():
            logger.info("Aggregating %s..." % stat_file)
            # Prep the data storage.
            my_agg = {}

            # Get a list of the relevant files (one per job).
            file_paths = file_tree.get_paths(stat_file)
            logger.info("Found %d files for %s." % (len(file_paths), stat_file))

            # Aggregate the data from all the jobs for each file type.
            for sub_path, file_entry in file_paths:
                s3_key = file_entry['key']
                ref = sub_path[0]
                file = s3.get_object(Bucket=bucket_name, Key=s3_key)
                file_bytes = file['Body'].read()
                if handle_stats is not None:
                    handle_stats(ref, my_agg, file_bytes)

            other_data = None
            if report_stats is not None and len(my_agg):
                other_data = report_stats(my_agg)

            stat_aggs[stat_file] = my_agg
            if other_data is not None:
                stat_aggs['gen_from_' + stat_file] = other_data

        for end_type, jobs in self.run_record.items():
            self.reporter.add_text('Jobs %s: %d' % (end_type, len(jobs)),
                                   section='Totals')

        s3_prefix = 'reading_results/%s/' % self.basename
        fname = self.reporter.make_report()
        with open(fname, 'rb') as f:
            s3.put_object(Bucket=bucket_name,
                          Key=s3_prefix + fname,
                          Body=f.read())
        s3.put_object(Bucket=bucket_name,
                      Key=s3_prefix + 'stat_aggregates_%s.pkl' % self.time_tag,
                      Body=pickle.dumps(stat_aggs))
        return file_tree, stat_aggs


def get_stage_choices(stage):
    if 'old_readings' in stage:
        label = 'old reading'
        c = 'cyan'
    elif 'new_readings' in stage:
        label = 'new reading'
        c = 'blue'
    elif 'make_statements' in stage:
        label = 'making statements'
        c = 'red'
    elif 'dump' in stage:
        label = 'dumping'
        c = 'black'
    elif stage == 'stats':
        label = 'generating stats'
        c = 'green'
    elif stage == 'jobs':
        label = 'total'
        c = 'grey'
    else:
        assert False, 'Unhandled stage: %s' % stage
    return label, c


if __name__ == '__main__':
    import argparse

    parent_submit_parser = create_submit_parser()
    parent_read_parser = create_read_parser()

    parser = argparse.ArgumentParser(
        'indra_db.reading.submig_reading_pipeline.py',
        parents=[parent_submit_parser, parent_read_parser],
        description=('Run reading with content on the db and submit results. '
                     'In this option, ids in \'input_file\' are given in the '
                     'as one text content id per line.'),
        )
    parser.add_argument(
        '-S', '--stmt_mode',
        choices=['all', 'unread', 'none'],
        default='all',
        help='Choose the subset of statements on which to run reading.'
    )
    parser.add_argument(
        '-R', '--reading_mode',
        choices=['all', 'unread', 'none'],
        default='unread',
        help=('Choose whether you want to read everything, nothing, or only '
              'the content that hasn\'t been read.')
    )
    parser.add_argument(
        '--max_reach_space_ratio',
        type=float,
        help='Set the maximum ratio of spaces to non-spaces for REACH input.',
        default=None
    )
    parser.add_argument(
        '--max_reach_input_len',
        type=int,
        help='Set the maximum length of content that REACH will read.',
        default=None
    )
    parser.add_argument(
        '--no_wait',
        action='store_true',
        help=('Don\'t run wait_for_complete at the end of the script. '
              'NOTE: wait_for_complete should always be run, so if it is not '
              'run here, it should be run manually.')
    )
    parser.add_argument(
        '--idle_log_timeout',
        type=int,
        default=600,
        help=("Set the time to wait for any given reading job to continue "
              "without any updates to the logs (at what point do you want it "
              "assumed dead).")
    )
    parser.add_argument(
        '--no_kill_on_timeout',
        action='store_true',
        help="If set, do not kill processes that have timed out."
    )
    args = parser.parse_args()

    sub = DbReadingSubmitter(args.basename, args.readers, args.project)
    sub.set_options(args.stmt_mode, args.reading_mode,
                    args.max_reach_input_len, args.max_reach_space_ratio)
    sub.submit_reading(args.input_file, args.start_ix, args.end_ix,
                       args.ids_per_job)
    if not args.now_wait:
        sub.watch_and_wait(idle_log_timeout=args.idle_log_timeout,
                           kill_on_timeout=not args.no_kill_on_timeout)
