# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:49:53 2019

@author: gbuster
"""
import os
import shutil
import itertools

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

        self._arg_combs, self._set_files = self._parse_config(self._config)

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
        set_files : list
            List of same length as arg_combs, representing the files to
            manipulate for each arg comb.
        """

        arg_combs = []
        set_files = []

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
                set_files.append(s['files'])

        return arg_combs, set_files

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
    def set_files(self):
        """List of files to be manipulated for each arg comb batch job.

        Returns
        -------
        set_files : list
            List of same length as arg_combs, representing the files to
            manipulate for each arg comb.
        """

        return self._set_files

    def _make_job_dirs(self):
        """Copy job files from the batch config dir into sub job dirs."""

        # walk through current directory getting everything to copy
        for dirpath, _, filenames in os.walk(self._base_dir):

            # For each dir level, iterate through the batch arg combinations
            for i, _ in enumerate(self.arg_combs):
                # tag and files to mod corresponding to this arg combination
                tag = self.job_tags[i]
                mod_files = self.set_files[i]
                mod_fnames = [os.path.basename(fn) for fn in mod_files]

                # Add the job tag to the directory path.
                # This will copy config subdirs into the job subdirs
                new_path = dirpath.replace(
                    self._base_dir, os.path.join(self._base_dir, tag + '/'))

                if not os.path.exists(new_path):
                    os.makedirs(new_path)

                for fn in filenames:

                    # straight copy
                    if fn not in mod_fnames:
                        shutil.copy(os.path.join(dirpath, fn),
                                    os.path.join(new_path, fn))


if __name__ == '__main__':
    fn = 'C:/sandbox/reV/git_reV2/examples/batched_execution/config_batch.json'
    b = BatchJob(fn)
    arg_combs = b.arg_combs
    set_files = b._set_files
    b._make_job_dirs()
