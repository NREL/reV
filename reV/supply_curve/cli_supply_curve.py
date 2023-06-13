# -*- coding: utf-8 -*-
"""
reV Supply Curve CLI utility functions.
"""
import logging
from warnings import warn

from gaps.cli import as_click_command, CLICommandFromClass
from gaps.pipeline import parse_previous_status

from reV.supply_curve.supply_curve import SupplyCurve
from reV.utilities.exceptions import PipelineError
from reV.utilities import ModuleName


logger = logging.getLogger(__name__)


def _preprocessor(config, out_dir):
    """Preprocess supply curve config user input.

    Parameters
    ----------
    config : dict
        User configuration file input as (nested) dict.
    out_dir : str
        Path to output file directory.

    Returns
    -------
    dict
        Updated config file.
    """
    if config.get("sc_points") == 'PIPELINE':
        sc_points = parse_previous_status(out_dir, ModuleName.SUPPLY_CURVE)
        if not sc_points:
            raise PipelineError('Could not parse "sc_points" from previous '
                                'pipeline jobs.')
        config["sc_points"] = sc_points[0]
        logger.info('Supply curve using the following '
                    'pipeline input for sc_points: {}'
                    .format(config["sc_points"]))

    if config.get("simple"):
        no_effect = [key for key in ['avail_cap_frac', 'line_limited']
                     if key in config]
        if no_effect:
            msg = ('The following key(s) have no effect when running '
                   'supply curve with "simple=True": "{}". To silence this '
                   'warning, please remove them from the config'
                   .format(', '.join(no_effect)))
            logger.warning(msg)
            warn(msg)

    return config


sc_command = CLICommandFromClass(SupplyCurve, method="run",
                                 name=str(ModuleName.SUPPLY_CURVE),
                                 add_collect=False, split_keys=None,
                                 config_preprocessor=_preprocessor)
main = as_click_command(sc_command)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV Supply Curve CLI.')
        raise
