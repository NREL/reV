# -*- coding: utf-8 -*-
"""
reV command line interface (CLI).
"""
import logging

from gaps.cli import make_cli
from reV.bespoke.cli_bespoke import bespoke_command
from reV.generation.cli_gen import gen_command
from reV.econ.cli_econ import econ_command
from reV.handlers.cli_collect import collect_command
from reV.handlers.cli_multi_year import my_command
from reV.supply_curve.cli_sc_aggregation import sc_agg_command
from reV.supply_curve.cli_supply_curve import sc_command
from reV.rep_profiles.cli_rep_profiles import rep_profiles_command
from reV.hybrids.cli_hybrids import hybrids_command
from reV.nrwal.cli_nrwal import nrwal_command
from reV.qa_qc.cli_qa_qc import qa_qc_command, qa_qc_extra
from reV.config.cli_project_points import project_points
from reV import __version__


logger = logging.getLogger(__name__)


commands = [bespoke_command, gen_command, econ_command, collect_command,
            my_command, sc_agg_command, sc_command, rep_profiles_command,
            hybrids_command, nrwal_command, qa_qc_command]
main = make_cli(commands, info={"name": "reV", "version": __version__})
main.add_command(qa_qc_extra)
main.add_command(project_points)

# export GAPs commands to namespace for documentation
batch = main.commands["batch"]
pipeline = main.commands["pipeline"]
script = main.commands["script"]
status = main.commands["status"]
reset_status = main.commands["reset-status"]
template_configs = main.commands["template-configs"]


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running reV CLI')
        raise
