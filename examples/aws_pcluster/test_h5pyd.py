import h5pyd

if __name__ == '__main__':
    fp = '/nrel/nsrdb/v3/nsrdb_2019.h5'
    with h5pyd.File(fp) as f:
        print(list(f))
