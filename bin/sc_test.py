"""
Full scale SupplyCurve Test
"""
import os
import pandas as pd
import time

from reV.handlers.transmission import TransmissionFeatures as TF
from reV.supply_curve.supply_curve import SupplyCurve
from reV.utilities.loggers import setup_logger, get_handler


def main():
    """
    Full scale supply curve test and timing
    """
    log_file = os.path.join(os.getcwd(), 'sc_test_2.log')
    logger = setup_logger('reV.supply_curve', log_file=log_file,
                          log_level="DEBUG")
    handler = get_handler()
    logger.addHandler(handler)

    try:
        path = '/scratch/ngrue/conus_trans_lines_cache_064_sj_infsink.csv'
        trans_table = pd.read_csv(path, index_col=0)

        path = '/scratch/gbuster/rev/test_sc_agg/agg.csv'
        sc_points = pd.read_csv(path, index_col=0)
        sc_points = sc_points.rename(columns={'sc_gid': 'sc_point_gid'})
        sc_points['sc_gid'] = sc_points.index.values

        ts = time.time()
        TF(trans_table)
        tt = time.time() - ts
        logger.info('Time to init TransmissionFeature = {:.4f} seconds'
                    .format(tt))

        ts = time.time()
        TF.feature_capacity(trans_table)
        tt = time.time() - ts
        logger.info('Time to compute available capacity = {:.4f} seconds'
                    .format(tt))

        ts = time.time()
        sc = SupplyCurve(sc_points, trans_table, fcr=0.096, connectable=False,
                         max_workers=36)
        tt = time.time() - ts
        logger.info('Time to init Simple SupplyCurve in parallel = {:.4f} '
                    'minutes'.format(tt / 60))

        ts = time.time()
        sc.simple_sort()
        tt = time.time() - ts
        logger.info('Time to run simple sort = {:.4f} minutes'.format(tt / 60))

        ts = time.time()
        sc = SupplyCurve(sc_points, trans_table, fcr=0.096, max_workers=36)
        tt = time.time() - ts
        logger.info('Time to init Full SupplyCurve in parallel = {:.4f} '
                    'minutes'.format(tt / 60))

        ts = time.time()
        sc.full_sort()
        tt = time.time() - ts
        logger.info('Time to run full sort = {:.4f} minutes'.format(tt / 60))

        ts = time.time()
        sc = SupplyCurve(sc_points, trans_table, fcr=0.096)
        tt = time.time() - ts
        logger.info('Time to init SupplyCurve in serial = {:.4f} minutes'
                    .format(tt / 60))

    except Exception:
        logger.exception('Error Running Test')
        raise


if __name__ == '__main__':
    main()
