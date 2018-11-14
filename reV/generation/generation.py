"""
Generation
"""
import logging
import os

import reV.SAM.SAM as SAM
from reV.config.config import Config
from reV import __dir__ as REVDIR
from reV.rev_logger import setup_logger
from reV.handlers import resource


class Gen:
    """Base class for generation"""
    def __init__(self, config_file):
        """Initialize a generation instance."""
        self._logger = logging.getLogger(self.__class__.__name__)

        _, handler = setup_logger(__name__)

        self._output_request = None
        self._config = Config(config_file)

        logger_list = ["reV.config", "reV.SAM", "reV.handlers"]
        loggers = {}
        for log in logger_list:
            loggers[log] = logging.getLogger(log)
            if not loggers[log].handlers:
                loggers[log].addHandler(handler)
            loggers[log].setLevel(self.config.logging_level)

        if not self._logger.handlers:
            self._logger.addHandler(handler)
        self._logger.setLevel(self.config.logging_level)

    @property
    def config(self):
        """Get the config object."""
        return self._config

    @property
    def output_request(self):
        """Get the list of output variables requested from generation."""
        if self._output_request is None:
            self._output_request = ['cf_mean']
            if self.config.SAM_gen.write_profiles:
                self._output_request += ['cf_profile']
        return self._output_request

    @property
    def project_points(self):
        """Get config project points"""
        return self._config.project_points

    def execute(self):
        """Execute the generation tech."""
        for file in self.config.res_files:

            if 'nsrdb' in file:
                res_iter = SAM.ResourceManager(resource.NSRDB(file),
                                               self.project_points,
                                               var_list=('dni', 'dhi',
                                                         'wind_speed',
                                                         'air_temperature'))
            elif 'wtk' in file:
                res_iter = SAM.ResourceManager(resource.WTK(file),
                                               self.project_points)

            if self.config.tech == 'pv':
                outputs = SAM.PV.reV_run(res_iter,
                                         self.project_points,
                                         output_request=self.output_request)

            elif self.config.tech == 'csp':
                outputs = SAM.CSP.reV_run(res_iter,
                                          self.project_points,
                                          output_request=self.output_request)

            elif self.config.tech == 'landbasedwind':
                outputs = SAM.LandBasedWind.reV_run(
                    res_iter, self.project_points,
                    output_request=self.output_request)

            elif self.config.tech == 'offshorewind':
                outputs = SAM.OffshoreWind.reV_run(
                    res_iter, self.project_points,
                    output_request=self.output_request)

            self._logger.debug('Outputs for {}: \n{}\n'
                               .format(file, outputs))

        return outputs


if __name__ == '__main__':
    # temporary script based test will be merged into test.py later
    cfile = os.path.join(REVDIR, 'config/ini/ri_subset_pv_gentest.ini')
    gen = Gen(cfile)
    config = gen.config
    gen._logger.setLevel(logging.DEBUG)
    outs = gen.execute()
    config = gen.config
    exec_control = gen.config.execution_control
    pp = gen.project_points
