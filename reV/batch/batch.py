# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:49:53 2019

@author: gbuster
"""
import copy
import json
import os
import shutil
import itertools

from reV.pipeline.pipeline import Pipeline
from reV.config.batch import BatchConfig


class BatchJob:
    """Framework for building a batched job suite."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : str
            File path to config json (str).
        """

        self._job_tags = None

        self._config = BatchConfig(config)
        self._base_dir = self._config.dir

        self._arg_combs, self._file_sets = self._parse_config(self._config)

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
            List of same length as arg_combs, representing the files to
            manipulate for each arg comb.
        """

        arg_combs = []
        file_sets = []

        # iterate through batch sets
        for s in config['sets']:

            # iterate through combinations of arg values
            for comb in itertools.product(*list(s['args'].values())):

                # make a dictionary representation of this combination
                comb_dict = {}
                for i, k in enumerate(s['args'].keys()):
                    comb_dict[k] = comb[i]

                # append the unique dictionary representation to the attr
                arg_combs.append(comb_dict)
                file_sets.append(s['files'])

        return arg_combs, file_sets

    def _make_job_tag(self, arg_comb):
        """Make a job tags from a unique combination of args + values.

        Returns
        -------
        job_tag : str
            Identifying string from the arg comb.
        """

        job_tag = []

        for arg, value in arg_comb.items():

            temp = arg.split('_')
            temp = ''.join([s[0] for s in temp])

            if isinstance(value, (int, str)):
                temp += str(value)

            elif isinstance(value, float):
                temp += str(value).replace('.', '')

            else:
                i = 0
                for s in self._config['sets']:
                    if arg in s['args']:
                        try:
                            i = s['args'][arg].index(value)
                        except ValueError as _:
                            pass
                        else:
                            break
                temp += str(i)

            job_tag.append(temp)

        job_tag = '_'.join(job_tag)

        return job_tag

    @property
    def job_tags(self):
        """Ordered list of job tags corresponding to unique arg/value combs.

        Returns
        -------
        _job_tags : list
            List of job tags corresponding to the unique arg/value
            combinations.
        """
        if self._job_tags is None:
            self._job_tags = []
            for arg_comb in self.arg_combs:
                self._job_tags.append(self._make_job_tag(arg_comb))
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
    def _mod_dict(inp, arg_mods):
        """Recursively modify key/value pairs in a dictionary.

        Parameters
        ----------
        inp : dict | list | str | int | float
            Input arg to modify. Should be a dict first, the recusive call
            can input nested values as dict/list/str etc...
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
                    out[k] = arg_mods[k]
                elif isinstance(v, (list, dict)):
                    out[k] = BatchJob._mod_dict(v, arg_mods)

        elif isinstance(inp, list):
            for i, entry in enumerate(inp):
                out[i] = BatchJob._mod_dict(entry, arg_mods)

        return out

    @staticmethod
    def _mod_json(fpath, fpath_out, arg_mods):
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

        with open(fpath, 'r') as f:
            data = json.load(f)

        data = BatchJob._mod_dict(data, arg_mods)

        with open(fpath_out, 'w') as f:
            json.dump(data, f, indent=4, separators=(',', ': '))

    def _make_job_dirs(self):
        """Copy job files from the batch config dir into sub job dirs."""

        # walk through current directory getting everything to copy
        for dirpath, _, filenames in os.walk(self._base_dir):

            # do make additional copies of job sub directories.
            skip = any([job_tag in dirpath for job_tag in self.job_tags])

            if not skip:

                # For each dir level, iterate through the batch arg combos
                for i, arg_comb in enumerate(self.arg_combs):
                    # tag and files to mod corresponding to this arg combo
                    tag = self.job_tags[i]
                    mod_files = self.file_sets[i]
                    mod_fnames = [os.path.basename(fn) for fn in mod_files]

                    # Add the job tag to the directory path.
                    # This will copy config subdirs into the job subdirs
                    new_path = dirpath.replace(
                        self._base_dir,
                        os.path.join(self._base_dir, tag + '/'))

                    if not os.path.exists(new_path):
                        os.makedirs(new_path)

                    for fn in filenames:

                        if fn in mod_fnames and fn.endswith('.json'):
                            # modify json and dump to new path
                            self._mod_json(os.path.join(dirpath, fn),
                                           os.path.join(new_path, fn),
                                           arg_comb)

                        else:
                            # straight copy of non-mod and non-json
                            shutil.copy(os.path.join(dirpath, fn),
                                        os.path.join(new_path, fn))

    def _run_pipelines(self):
        """Run the reV pipeline modules for each batch job."""
        for d in self.sub_dirs:
            config_pipeline = os.path.join(
                d, os.path.basename(self._config.config_pipeline))
            Pipeline.run(config_pipeline, monitor=False)

    @classmethod
    def run(cls, config):
        """Run the reV batch job from a config file.

        Parameters
        ----------
        config : str
            File path to config json (str).
        """

        b = cls(config)
        b._make_job_dirs()
        b._run_pipelines()
