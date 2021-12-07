# -*- coding: utf-8 -*-
# pylint: skip-file
"""reV SAM unit test module
"""
import json
import os
import shutil
import tempfile

from reV import TESTDATADIR
from reV.bespoke.bespoke import BespokeWindFarms
from reV.supply_curve.tech_mapping import TechMapping

from rex import init_logger


SAM = os.path.join(TESTDATADIR, 'SAM/i_windpower.json')
EXCL = os.path.join(TESTDATADIR, 'ri_exclusions/ri_exclusions.h5')
RES = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_{}.h5')
TM_DSET = 'techmap_wtk_ri_100'
AGG_DSET = ('cf_mean', 'cf_profile')

# note that this differs from the
EXCL_DICT = {'ri_srtm_slope': {'inclusion_range': (None, 5),
                               'exclude_nodata': False},
             'ri_padus': {'exclude_values': [1],
                          'exclude_nodata': False},
             'ri_reeds_regions': {'inclusion_range': (None, 400),
                                  'exclude_nodata': False}}


if __name__ == '__main__':
    init_logger('reV', log_level='DEBUG')
    gid = 33  # 39% included
    ws_dset = 'windspeed_88m'
    wd_dset = 'winddirection_88m'

    with open(SAM, 'r') as f:
        sam_sys_inputs = json.load(f)

    with tempfile.TemporaryDirectory() as td:
        excl_fp = os.path.join(td, 'ri_exclusions.h5')
        res_fp = os.path.join(td, 'ri_100_wtk_{}.h5')
        shutil.copy(EXCL, excl_fp)
        shutil.copy(RES.format(2012), res_fp.format(2012))
        shutil.copy(RES.format(2013), res_fp.format(2013))
        res_fp = res_fp.format('*')

        TechMapping.run(excl_fp, RES.format(2012), dset=TM_DSET, max_workers=1)
        BespokeWindFarms.run_serial(excl_fp, res_fp, TM_DSET, ws_dset, wd_dset,
                                    sam_sys_inputs,
                                    excl_dict=EXCL_DICT, gids=gid)
