"""
RPM CLI entry points.
"""
import click
import logging
import os
import pprint
import time
import json



logger = logging.getLogger(__name__)


def main(hdf_ws, resource, title, year, floor, verbose, initial, indices, calc_p, weight_type,
         region_column, output_ws, wavelet, in_db, out_db, meta, region, save_to_disk, output_name,
         desc, final_tz):

    # get relevant gid values
    _gids = get_gids_by_region(region=region, column=region_column, limit=None, **in_db)
    LOGGER.info('[{r}] clustering {p:d} profiles'.format(p=_gids.shape[0], r=region))

    # get relevant chunks
    _chunks = get_chunks(df=_gids, meta=meta)

    # subset metadata by chunk values relevant to gids
    meta = meta[['gid', 'chunk', 'tz']][np.in1d(meta['chunk'].values, _chunks)]

    _start_time = datetime.now()

   # @TODO need better time_intervals detection
    _time_intervals = 8760 if resource in ('wind', ) else 17520

    # get relevant profiles
    _raw_profiles, _profiles_stable = get_profiles(hdf_ws=hdf_ws,
                                                   chunks=_chunks,
                                                   gids=_gids['gid'].values,
                                                   meta=meta,
                                                   resource=resource,
                                                   title=title,
                                                   year=year,
                                                   region=region,
                                                   time_intervals=_time_intervals,
                                                   utc_offset=final_tz)

    # subset _gids by gid values available in project
    _gids = _gids[np.in1d(_gids['gid'].values, _raw_profiles['gid'])]

    if _profiles_stable.shape[0] < 1:
        LOGGER.warning('no profiles for {} were found in {}'.format(region, hdf_ws))
        sys.exit(0)

    # calculate wavelet coefficients
    _coefficients = get_dwt_coefficients(x=_profiles_stable.drop('gid', axis=1).values,
                                         wavelet=wavelet, level=None, indices=indices,
                                         region=region)

    # prepare weight matrix
    _w, _profiles_stable, _islands = get_weight_matrix(gdf=_gids, profiles=_profiles_stable,
                                                       weight_type=weight_type, region=region)

    # get regions
    _maxp = cluster_maxp(z=_coefficients, weight=_w, floor=floor, verbose=verbose,
                         initial=initial, p=calc_p, region=region)

    # @TODO: need way to reintegrate island pixels; currently drop them as they occur, could instead
    #  use distance weight matrix if data are or could be projected

    # post-process regions
    try:
        LOGGER.info('[{r}] created {n:d} regions at level {i:d}'.format(r=region,
                                                                        n=len(_maxp.regions),
                                                                        i=0))
    except AttributeError:
        sys.exit('no initial solution found; try decreasing <floor> or including more profiles')
    else:
        _regions, _rid_mapping = post_process_maxp(x=_maxp, gids=_gids, orig_profiles=_raw_profiles,
                                                   output_ws=output_ws, region_name=region,
                                                   weight_type=weight_type,
                                                   coefficient_indices=indices, floor=floor,
                                                   initial=initial, iteration=0, out_db=out_db,
                                                   in_db=in_db, save_to_disk=save_to_disk,
                                                   resource_type=resource, output_name=output_name,
                                                   desc=desc, title=title,
                                                   raw_profiles=_raw_profiles.set_index('gid'),
                                                   rid_mapping=None)

        _time_elapsed = datetime.now() - _start_time
        LOGGER.info('[{r}] clustered {p:d} profiles into {g:d} regions '
                    'at level {i:d} ({h})'.format(p=_gids.shape[0], g=_regions.shape[0], i=0,
                                                  h=_time_elapsed, r=region))

    # loop results until we can't find any more solutions
    _iteration = 1
    while _regions.shape[0] >= floor:
        _start_time = datetime.now()
        # create new gids dataframe
        _gids = _regions[['gid', 'geom']]

        # select out profiles
        _profiles = _regions.drop(['geom', 'ann_cf_mean'], axis=1)

        # get new stable profiles
        _profiles_stable = stabilize(x=_profiles.drop('gid', axis=1).values)
        _profiles_stable = pd.DataFrame(data=_profiles_stable)
        _profiles_stable['gid'] = _profiles['gid']

        # get new coefficients
        _coefficients = get_dwt_coefficients(x=_profiles_stable, wavelet=wavelet, level=None,
                                             indices=indices, region=region)
        # prepare new weight matrix
        _w, _profiles_stable, _islands = get_weight_matrix(gdf=_gids, profiles=_profiles_stable,
                                                           weight_type=weight_type, region=region)
        # get new regions
        _maxp = cluster_maxp(z=_coefficients, weight=_w, floor=floor, verbose=verbose,
                             initial=initial, p=calc_p, region=region)
        try:
            LOGGER.info('[{r}] created {n:d} regions at level {i:d}'.format(r=region,
                                                                            n=len(_maxp.regions),
                                                                            i=_iteration))
        except AttributeError:
            LOGGER.info('[{r}] No more solutions available'.format(r=region))
            break
        else:
            # post-process new regions
            _regions, _rid_mapping = post_process_maxp(x=_maxp, gids=_gids, orig_profiles=_profiles,
                                                       output_ws=output_ws, region_name=region,
                                                       weight_type=weight_type,
                                                       coefficient_indices=indices, floor=floor,
                                                       initial=initial, iteration=_iteration,
                                                       out_db=out_db, in_db=in_db,
                                                       save_to_disk=save_to_disk, desc=desc,
                                                       output_name=output_name,
                                                       resource_type=resource, title=title,
                                                       raw_profiles=_raw_profiles.set_index('gid'),
                                                       rid_mapping=_rid_mapping)

        _kvals = {'profiles_in': _gids.shape[0],
                  'regions_out': _regions.shape[0],
                  'iterations': _iteration,
                  'time_elapsed': _time_elapsed}

        _time_elapsed = datetime.now() - _start_time
        LOGGER.info('[{r}] clustered {profiles_in:d} profiles into {regions_out:d} regions'
                    ' at level {iterations:d} ({time_elapsed})'.format(r=region, **_kvals))

        _iteration += 1

    return region


def _main(kvals):
    main(**kvals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='profiles.py',
                                     description='Collect representative profiles for resource regions',
                                     fromfile_prefix_chars='@')
    parser.add_argument('rpm_project', type=str, default=None, help='RPM project schema')
    rev_group = parser.add_argument_group('reV project arguments')
    rev_group.add_argument('rev_project', type=str, default=None, help='reV project output root directory')
    rev_group.add_argument('rev_resource', type=str, default=None, help='reV project resource name of source profiles, e.g., pv, wind')
    rev_group.add_argument('--rev_year', type=int, default=2012, help='reV output year (default: %(default)s)')

    region_group = parser.add_argument_group('resource region arguments')
    region_group.add_argument('region_schema', type=str, default=None, help='schema containing <region_table>, i.e., RPM project schema')
    region_group.add_argument('region_table', type=str, default=None, help='table with cluster IDs, i.e., cluster table')
    region_group.add_argument('region_resource', type=str, default=None, help='resource name for clusters , e.g., wind, pv_1axis, pv_tilt, csp')
    region_group.add_argument('--grid_id', dest='grid_id', default='grid_gid_v3', help='resource grid ID column name (default: %(default)s)')

    resource_group = parser.add_argument_group('resource arguments')
    resource_group.add_argument('--resource_schema', dest='resource_schema', type=str, default='nsrdb', help='schema containing <resource_table> (default: %(default)s)')
    resource_group.add_argument('--resource_table', dest='resource_table', type=str, default='gid_lookup_join', help='table containing gridded resource data with column <resource_id> and <region_column) (default: %(default)s)')
    resource_group.add_argument('--region_column', dest='region_column', type=str, default='ba', help='column in <resource_table> denoting resource region grid cell belongs to (default: %(default)s)')
    resource_group.add_argument('--resource_id', dest='resource_id', type=str, default='gid_v3', help='resource grid identifier (default: %(default)s)')

    exclusion_group = parser.add_argument_group('exclusion options')
    exclusion_group.add_argument('--exclusion_schema', dest='exclusion_schema', type=str, default=None, help='schema containing <exclusion_table>')
    exclusion_group.add_argument('--exclusion_table', dest='exclusion_table', type=str, default=None, help='table containing <grid_id> and inc_pct columns')
    exclusion_group.add_argument('--inclusion_pct', dest='inclusion_pct', type=float, default=None, help='minimum required included area')

    database_group = parser.add_argument_group('database connection options')
    database_group.add_argument('--db_host', dest='db_host', type=str, default='gispgdb.nrel.gov', help='default: %(default)s')
    database_group.add_argument('--db_name', dest='db_name', type=str, default='dav-gis', help='default: %(default)s')

    output_group = parser.add_argument_group('output options')
    output_group.add_argument('--utc_offset', dest='utc_offset', type=int, default=-7)

    _args = parser.parse_args()

    _project = _args.rpm_project #'rpm_la'  # _args.rpm_project

    _year = _args.rev_year #2012  # _args.year

    _grid_id = _args.grid_id #'grid_gid_v3'  # _args.grid_id
    _region_resource = _args.region_resource  #'pv_1axis'  # _args.region_resource
    _profile_resource = _args.rev_resource  #'pv_tilt'  # _args.rev_resource
    _region_schema = _args.region_schema
    _region_table = _args.region_table # = 'solar_fixed_tilt_clusters_profile_test'  # '"%s" % '"."'.join(_args.region_schema, _args.region_table)
    _rev_project_ws = _args.rev_project #'/projects/rev/projects/nsrdbv3_gen_fixlat_inv13_profs_pv_conus_2012/'  # _args.rev_project
    _rev_resource = _args.rev_resource #'pv'  # _args.rev_resource

    _resource_schema = _args.resource_schema
    _resource_table = _args.resource_table
    _region_column = _args.region_column
    _resource_id = _args.resource_id

    _trg = _region_resource in ('wind', 'offshore')

    _exc_schema = _args.exclusion_schema
    _exc_table = _args.exclusion_table
    _inc_pct = _args.inclusion_pct

    _db_host = _args.db_host  #'gispgdb.nrel.gov'  # _args.db_host
    _db_name = _args.db_name  # 'dav-gis'  # _args.db_name

    _utc_offset = _args.utc_offset  # -7

    # get meta
    _meta = get_metadata(fpath=os.path.join(_rev_project_ws, 'scalar_outputs', 'project_outputs.h5'), resource=_rev_resource)

    # get all regions
    _regions = get_regions(db_host=_db_host, db_name=_db_name, schema=_resource_schema, table=_resource_table, column=_region_column)
    #_regions = _regions.loc[_regions.region == 'DE']
    LOGGER.debug(_regions)

    # get cluster level
    _regions['cluster_level'] = _regions.region.apply(get_cluster_level, region_resource=_region_resource, project=_project, db_host=_db_host, db_name=_db_name)

    # init non-null clusters index
    _nonnull_index = pd.notnull(_regions).all(1).nonzero()

    # get grid GIDs and cluster IDs
    _regions['clusters'] = _regions.iloc[_nonnull_index].apply(func=lambda x: get_cluster_ids(*x.values, schema=None, table=None, exc_schema=_exc_schema, exc_table=_exc_table, inc_pct=_inc_pct, grid_id=_grid_id, region_resource=_region_resource, trg=_trg, db_host=_db_host, db_name=_db_name), axis=1)  # @TODO: needs exc_schema, exc_table, exc_pct, grid_id, region_resource, trg, db_host, db_name
    # get GID of representative profiles for each cluster
    _regions_rep_gids = _regions.iloc[_nonnull_index][['region', 'clusters']].apply(func=lambda x, meta: get_rep_profile(*x, meta=meta, cluster_schema=_region_schema, cluster_table=_region_table, year=_year, profile_resource=_profile_resource, rev_project_ws=os.path.join(_rev_project_ws, 'profile_outputs'), rev_resource=_rev_resource, db_host=_db_host, db_name=_db_name, utc_offset=_utc_offset), axis=1, meta=_meta)

    LOGGER.info('complete')
