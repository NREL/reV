# -*- coding: utf-8 -*-
"""reV batching framework for parametric runs.

The batch framework allows users to modify key-value pairs in input jsons files
based on a batch config file. The batch module will create run directories for
all combinations of input parametrics, and run the reV pipelines for each job.

Created on Mon Jun 10 13:49:53 2019

@author: gbuster
"""
import pandas as pd
import copy
import json
import os
import shutil
import itertools
import logging
from warnings import warn

from reV.pipeline.pipeline import Pipeline
from reV.config.batch import BatchConfig
from reV.utilities.exceptions import PipelineError
from reV.pipeline.cli_pipeline import pipeline_monitor_background

from rex.utilities import safe_json_load, parse_year
from rex.utilities.loggers import init_logger


logger = logging.getLogger(__name__)


class BatchJob:
    """Framework for building a batched job suite."""

    def __init__(self, config, verbose=False):
        """
        Parameters
        ----------
        config : str
            File path to config json or csv (str).
        verbose : bool
            Flag to turn on debug logging.
        """

        self._job_tags = None

        self._config = BatchConfig(config)
        self._base_dir = self._config.config_dir
        os.chdir(self._base_dir)

        if 'logging' in self._config:
            logging_kwargs = self._config.get('logging', {})
            if verbose:
                logging_kwargs['log_level'] = 'DEBUG'

            init_logger('reV.batch', **logging_kwargs)

        x = self._parse_config(self._config)
        self._arg_combs, self._file_sets, self._set_tags = x

        logger.info('Batch job initialized with {} sub jobs.'
                    .format(len(self._arg_combs)))

    @staticmethod
    def _parse_config(config):
        """Parse batch config object for useful data.

        Parameters
        ----------
        config : from reV.config.batch.BatchConfig
            reV batch config object that emulates a dict.

        Returns
        -------
        arg_combs : list
            List of dictionaries representing the different arg/value
            combinations made available in the batch config json.
        file_sets : list
            List representing the files to manipulate for each arg comb
            (same length as arg_combs).
        set_tags : list
            List of strings of tags for each batch job set
            (same length as arg_combs).
        """

        arg_combs = []
        file_sets = []
        set_tags = []
        sets = []

        # iterate through batch sets
        for s in config['sets']:
            set_tag = s.get('set_tag', '')
            if set_tag in sets:
                msg = ('Found multiple sets with the same set_tag: "{}"'
                       .format(set_tag))
                logger.error(msg)
                raise ValueError(msg)
            else:
                sets.append(set_tag)

            # iterate through combinations of arg values
            for comb in itertools.product(*list(s['args'].values())):

                # make a dictionary representation of this combination
                comb_dict = {}
                for i, k in enumerate(s['args'].keys()):
                    comb_dict[k] = comb[i]

                # append the unique dictionary representation to the attr
                arg_combs.append(comb_dict)
                file_sets.append(s['files'])
                set_tags.append(set_tag)

        return arg_combs, file_sets, set_tags

    @staticmethod
    def _fix_tag_w_year(value):
        """If one of the tag values looks like a year, add a zero for the tag.

        Parameters
        ----------
        value : str | int | float
            Value from a batch run.

        Returns
        -------
        value : str
            Value from a batch run converted to a string tag, if the input
            value looks like a year, a zero will be added.
        """

        value = str(value).replace('.', '')

        if parse_year('_' + value, option='bool'):
            value += '0'

        return value

    def _make_job_tag(self, set_tag, arg_comb):
        """Make a job tags from a unique combination of args + values.

        Parameters
        ----------
        set_tag : str
            Optional set tag to prefix job tag.
        arg_comb : dict
            Key-value pairs for this argument combination.

        Returns
        -------
        job_tag : str
            Identifying string from the arg comb.
        """

        job_tag = []

        for arg, value in arg_comb.items():
            temp = arg.split('_')
            temp = ''.join([s[0] for s in temp])

            if isinstance(value, (int, float)):
                temp += self._fix_tag_w_year(value)

            else:
                i = 0
                for s in self._config['sets']:
                    if arg in s['args']:
                        try:
                            i = s['args'][arg].index(value)
                        except ValueError:
                            pass
                        else:
                            break
                temp += str(i)

            job_tag.append(temp)

        job_tag = '_'.join(job_tag)

        if set_tag:
            job_tag = '{}_{}'.format(set_tag, job_tag)

        if job_tag.endswith('_'):
            job_tag = job_tag.rstrip('_')

        return job_tag

    def _clean_arg_comb_tag(self, set_tag, arg_comb):
        """Clean a dictionary of arg combinations for a single job by removing
        any args that only have one value in the current set tag.

        Parameters
        ----------
        set_tag : str
            Optional set tag to prefix job tag.
        arg_comb : dict
            Key-value pairs for this argument combination.

        Returns
        -------
        tag_arg_comb : str
            Arg combinations just for making the job tags. This may not have
            all the arg combinations for the actual job setup.
        """

        ignore_tags = []

        for arg in arg_comb.keys():
            all_values = []
            for batch_set in self._config['sets']:
                if (batch_set.get('set_tag', '') == set_tag
                        and arg in batch_set['args']):
                    all_values += batch_set['args'][arg]

            if len(all_values) <= 1:
                ignore_tags.append(arg)

        tag_arg_comb = {k: v for k, v in arg_comb.items()
                        if k not in ignore_tags}

        return tag_arg_comb

    @property
    def job_table(self):
        """Get a dataframe summarizing the batch jobs."""
        table = pd.DataFrame()
        for i, job_tag in enumerate(self.job_tags):
            job_info = {k: str(v) for k, v in self.arg_combs[i].items()}
            job_info['set_tag'] = str(self._set_tags[i])
            job_info['files'] = str(self.file_sets[i])
            job_info = pd.DataFrame(job_info, index=[job_tag])
            table = table.append(job_info)

        table.index.name = 'job'

        return table

    @property
    def job_tags(self):
        """Ordered list of job tags corresponding to unique arg/value combs.

        Returns
        -------
        job_tags : list
            List of job tag strings corresponding to the unique arg/value
            combinations.
        """
        if self._job_tags is None:
            self._job_tags = []
            for i, arg_comb in enumerate(self.arg_combs):
                tag_arg_comb = self._clean_arg_comb_tag(self._set_tags[i],
                                                        arg_comb)
                self._job_tags.append(self._make_job_tag(self._set_tags[i],
                                                         tag_arg_comb))
        return self._job_tags

    @property
    def arg_combs(self):
        """List of all viable arg/value combinations.

        Returns
        -------
        _arg_combs : list
            List of dictionaries representing the different arg/value
            combinations made available in the batch config json.
        """

        return self._arg_combs

    @property
    def file_sets(self):
        """List of files to be manipulated for each arg comb batch job.

        Returns
        -------
        file_sets : list
            List of same length as arg_combs, representing the files to
            manipulate for each arg comb.
        """

        return self._file_sets

    @property
    def sub_dirs(self):
        """List of job sub directories.

        Returns
        -------
        sub_dirs : list
            List of strings of job sub directories.
        """
        sub_dirs = [os.path.join(self._base_dir, tag + '/')
                    for tag in self.job_tags]
        return sub_dirs

    @staticmethod
    def _clean_arg(arg):
        """Perform any cleaning steps required before writing an arg to a json

        Cleaning steps:
            1. Convert stringified dictionary to python dictionary object

        Parameters
        ----------
        arg : str | int | float | dict
            Value to be written to a batch job json file

        Returns
        -------
        arg : str | int | float | dict
            Cleaned value ready to be written to a batch job json file.
        """

        if isinstance(arg, str):
            if '{' in arg and '}' in arg and ("'" in arg or '"' in arg):
                arg = arg.replace("'", '"')
                try:
                    arg = json.loads(arg)
                except json.decoder.JSONDecodeError as e:
                    msg = 'Could not load json string: {}'.format(arg)
                    logger.exception(msg)
                    raise e

        return arg

    @classmethod
    def _mod_dict(cls, inp, arg_mods):
        """Recursively modify key/value pairs in a dictionary.

        Parameters
        ----------
        inp : dict | list | str | int | float
            Input to modify such as a loaded json reV config to modify with
            batch. Can also be nested data from a recursive call. Should be a
            dict first, the recusive call can input nested values as
            dict/list/str etc...
        arg_mods : dict
            Key/value pairs to insert in the inp.

        Returns
        -------
        out : dict | list | str | int | float
            Modified inp with arg_mods.
        """

        out = copy.deepcopy(inp)

        if isinstance(inp, dict):
            for k, v in inp.items():
                if k in arg_mods:
                    out[k] = cls._clean_arg(arg_mods[k])
                elif isinstance(v, (list, dict)):
                    out[k] = cls._mod_dict(v, arg_mods)

        elif isinstance(inp, list):
            for i, entry in enumerate(inp):
                out[i] = cls._mod_dict(entry, arg_mods)

        return out

    @classmethod
    def _mod_json(cls, fpath, fpath_out, arg_mods):
        """Import and modify the contents of a json. Dump to new file.

        Parameters
        ---------
        fpath : str
            File path to json to be imported/modified
        fpath_out : str
            File path to dump new modified json.
        arg_mods : dict
            Dictionary representing one set of arg/value combinations to
            implement in the json
        """

        data = safe_json_load(fpath)
        data = cls._mod_dict(data, arg_mods)

        with open(fpath_out, 'w') as f:
            json.dump(data, f, indent=4, separators=(',', ': '))

    def _make_job_dirs(self):
        """Copy job files from the batch config dir into sub job dirs."""

        table = self.job_table
        table.to_csv(os.path.join(self._base_dir, 'batch_jobs.csv'))
        logger.debug('Batch jobs list: {}'
                     .format(sorted(table.index.values.tolist())))
        logger.info('Preparing batch job directories...')

        # walk through current directory getting everything to copy
        for source_dir, _, filenames in os.walk(self._base_dir):

            # do make additional copies of job sub directories.
            skip = any([job_tag in source_dir for job_tag in self.job_tags])

            if not skip:

                # For each dir level, iterate through the batch arg combos
                for i, arg_comb in enumerate(self.arg_combs):
                    # tag and files to mod corresponding to this arg combo
                    tag = self.job_tags[i]
                    mod_files = self.file_sets[i]
                    mod_fnames = [os.path.basename(fn) for fn in mod_files]

                    # Add the job tag to the directory path.
                    # This will copy config subdirs into the job subdirs
                    destination_dir = source_dir.replace(
                        self._base_dir,
                        os.path.join(self._base_dir, tag + '/'))

                    if not os.path.exists(destination_dir):
                        logger.debug('Making new job directory: {}'
                                     .format(destination_dir))
                        os.makedirs(destination_dir)

                    for fn in filenames:
                        self._copy_batch_file(fn, source_dir, destination_dir,
                                              tag, arg_comb, mod_fnames)

        logger.info('Batch job directories ready for execution.')

    def _copy_batch_file(self, fn, source_dir, destination_dir, job_name,
                         arg_comb, mod_fnames):
        """Copy a file in the batch directory into a job directory with
        appropriate batch modifications (permutations) as defined by arg_comb.

        Parameters
        ----------
        fn : str
            Filename being copied.
        source_dir : str
            Directory that the file is sourced from (usually the base batch
            project directory)
        destination_dir : str
            Directory that the file should be copied to (usually a job's
            sub directory in source_dir)
        job_name : str
            Batch job name (for logging purposes)
        arg_comb : dict
            Dictionary representing one set of arg/value combinations to
            implement in the fn if the fn is a .json file specified as one
            of the batch permutation files to modify.
        mod_fnames : list
            List of filenames that need to be modified by the batch module.
        """

        if fn in mod_fnames and fn.endswith('.json'):
            # modify json and dump to new path
            logger.debug('Copying and modifying run json file '
                         '"{}" to job: "{}"'.format(fn, job_name))
            self._mod_json(os.path.join(source_dir, fn),
                           os.path.join(destination_dir, fn),
                           arg_comb)

        else:
            # straight copy of non-mod and non-json
            fp_source = os.path.join(source_dir, fn)
            fp_target = os.path.join(destination_dir, fn)

            modified = True
            if os.path.exists(fp_target):
                modified = (os.path.getmtime(fp_source)
                            > os.path.getmtime(fp_target))

            if modified:
                logger.debug('Copying run file "{}" to job: '
                             '"{}"'.format(fn, job_name))
                try:
                    shutil.copy(fp_source, fp_target)
                except Exception:
                    msg = ('Could not copy "{}" to job: "{}"'
                           .format(fn, job_name))
                    logger.warning(msg)
                    warn(msg)

    def _run_pipelines(self, monitor_background=False, verbose=False):
        """Run the reV pipeline modules for each batch job.

        Parameters
        ----------
        monitor_background : bool
            Flag to monitor all batch pipelines continuously
            in the background using the nohup command. Note that the
            stdout/stderr will not be captured, but you can set a
            pipeline "log_file" to capture logs.
        verbose : bool
            Flag to turn on debug logging for the pipelines.
        """

        for d in self.sub_dirs:
            pipeline_config = os.path.join(
                d, os.path.basename(self._config.pipeline_config))
            if not os.path.isfile(pipeline_config):
                raise PipelineError('Could not find pipeline config to run: '
                                    '"{}"'.format(pipeline_config))
            elif monitor_background:
                pipeline_monitor_background(pipeline_config, verbose=verbose)
            else:
                Pipeline.run(pipeline_config, monitor=False, verbose=verbose)

    def _cancel_all(self):
        """Cancel all reV pipeline modules for all batch jobs."""
        for d in self.sub_dirs:
            pipeline_config = os.path.join(
                d, os.path.basename(self._config.pipeline_config))
            if os.path.isfile(pipeline_config):
                Pipeline.cancel_all(pipeline_config)

    def _delete_all(self):
        """Clear all of the batch sub job folders based on the job summary
        csv file in the batch config directory."""

        fp_job_table = os.path.join(self._base_dir, 'batch_jobs.csv')
        if not os.path.exists(fp_job_table):
            msg = ('Cannot delete batch jobs without jobs summary table: {}'
                   .format(fp_job_table))
            logger.error(msg)
            raise FileNotFoundError(msg)

        job_table = pd.read_csv(fp_job_table, index_col=0)

        if job_table.index.name != 'job':
            msg = ('Cannot delete batch jobs when the batch summary table '
                   'does not have "job" as the index key')
            logger.error(msg)
            raise ValueError(msg)

        for sub_dir in job_table.index:
            job_dir = os.path.join(self._base_dir, sub_dir)
            if os.path.exists(job_dir):
                logger.info('Removing batch job directory: {}'.format(sub_dir))
                shutil.rmtree(job_dir)
            else:
                w = 'Cannot find batch job directory: {}'.format(sub_dir)
                logger.warning(w)
                warn(w)

        os.remove(fp_job_table)

    @classmethod
    def cancel_all(cls, config, verbose=False):
        """Cancel all reV pipeline modules for all batch jobs.

        Parameters
        ----------
        config : str
            File path to batch config json or csv (str).
        verbose : bool
            Flag to turn on debug logging.
        """

        b = cls(config, verbose=verbose)
        b._cancel_all()

    @classmethod
    def delete_all(cls, config, verbose=False):
        """Delete all reV batch sub job folders based on the job summary csv
        in the batch config directory.

        Parameters
        ----------
        config : str
            File path to batch config json or csv (str).
        verbose : bool
            Flag to turn on debug logging.
        """
        b = cls(config, verbose=verbose)
        b._delete_all()

    @classmethod
    def run(cls, config, dry_run=False, delete=False, monitor_background=False,
            verbose=False):
        """Run the reV batch job from a config file.

        Parameters
        ----------
        config : str
            File path to config json or csv (str).
        dry_run : bool
            Flag to make job directories without running.
        delete : bool
            Flag to delete all batch job sub directories based on the job
            summary csv in the batch config directory.
        monitor_background : bool
            Flag to monitor all batch pipelines continuously
            in the background using the nohup command. Note that the
            stdout/stderr will not be captured, but you can set a
            pipeline "log_file" to capture logs.
        verbose : bool
            Flag to turn on debug logging for the pipelines.
        """

        b = cls(config, verbose=verbose)
        if delete:
            b._delete_all()
        else:
            b._make_job_dirs()
            if not dry_run:
                b._run_pipelines(monitor_background=monitor_background,
                                 verbose=verbose)
