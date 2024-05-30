"""
Make small test project points for aws pcluster example.
"""
from rex import Resource

if __name__ == '__main__':
    fp = '/nrel/nsrdb/v3/nsrdb_2019.h5'
    fp = '/nrel/wtk/conus/wtk_conus_2007.h5'

    lat_range = (39.33, 40.29)
    lon_range = (-107.46, -104.54)

    with Resource(fp, hsds=True) as res:
        meta = res.meta

    mask = ((meta.longitude >= lon_range[0])
            & (meta.longitude <= lon_range[1])
            & (meta.latitude >= lat_range[0])
            & (meta.latitude <= lat_range[1])
            )

    print(meta[mask])
    pp = meta[mask]
    pp["gid"] = pp.index.values
    pp['config'] = 'def'
    pp = pp[["gid", 'config']]
    pp.to_csv('./points_front_range.csv', index=False)
