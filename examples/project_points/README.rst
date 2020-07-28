reV Project Points
==================

`reV Gen <https://nrel.github.io/reV/reV/reV.generation.generation.html#reV.generation.generation.Gen>`_
and `reV Econ <https://nrel.github.io/reV/reV/reV.econ.econ.html#reV.econ.econ.Econ>`_
use `Project Points <https://nrel.github.io/reV/reV/reV.config.project_points.html#reV.config.project_points.ProjectPoints>`_ to define which resource sites (`gids`) to run through
`PySAM <https://pysam.readthedocs.io/en/latest/>`_ and how.

At its most basic Project Points consists of the resource ``gid``s and the
``SAM`` configuration file associated it. This can be definited in a variety
of ways:

1) From a project points .csv and a single or dictionary of ``SAM``
   configuration files:

.. code-block:: python

    import os
    from reV import TESTDATADIR
    from reV.config.project_points import ProjectPoints

    fpp = os.path.join(TESTDATADIR, 'project_points/pp_offshore.csv')
    sam_files = {'onshore': os.path.join(
                 TESTDATADIR, 'SAM/wind_gen_standard_losses_0.json'),
                 'offshore': os.path.join(
                 TESTDATADIR, 'SAM/wind_gen_standard_losses_1.json')}

    pp = ProjectPoints(fpp, sam_files)
    display(pp.df)

                gid   config
    0       2114919  onshore
    1       2114920  onshore
    2       2114921  onshore
    3       2114922  onshore
    4       2114923  onshore
    ...         ...      ...
    124402  2239321  onshore
    124403  2239322  onshore
    124404  2239323  onshore
    124405  2239324  onshore
    124406  2239325  onshore

    [124407 rows x 2 columns]

2) From a list or slice of gids and a single ``SAM`` configuration file:

.. code-block:: python

    import os
    from reV import TESTDATADIR
    from reV.config.project_points import ProjectPoints

    sites = slice(0, 100)  # or
    sites = [0, 5, 6, 9, 12]

    sam_file = os.path.join(TESTDATADIR, 'SAM/wind_gen_standard_losses_0.json')

    pp = ProjectPoints(sites, sam_file)
    display(pp.df)

       gid                                             config
    0    0  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    1    5  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    2    6  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    3    9  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    4   12  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...

3) From a pair or pairs of latitude and longitude coordinates and a single
   ``SAM`` configuration file (NOTE: access to the resource file to be used
   for ``reV Gen`` or ``reV Econ`` is needed to find the associated resource
   gids):

.. code-block:: python

    import os
    from reV import TESTDATADIR
    from reV.config.project_points import ProjectPoints

    lat_lons = [41.77, -71.74]
    lat_lons = array([[ 41.77, -71.74],
                      [ 41.73, -71.7 ],
                      [ 42.01, -71.7 ],
                      [ 40.97, -71.74],
                      [ 41.49, -71.78]])

    res_file = os.path.join(TESTDATADIR, 'nsrdb/', 'ri_100_nsrdb_2012.h5')
    sam_file = os.path.join(TESTDATADIR, 'SAM/wind_gen_standard_losses_0.json')

    pp = ProjectPoints.lat_lon_coords(lat_lons, res_file, sam_file)
    display(pp.df)

       gid                                             config
    0   49  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    1   67  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    2   79  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    3   41  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    4   31  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...

4) A geographic region or regions and a single ``SAM`` configuration file
   (NOTE: access to the resource file to be used for ``reV Gen`` or
   ``reV Econ`` is needed to find the associated resource gids):

.. code-block:: python

    import os
    from reV import TESTDATADIR
    from reV.config.project_points import ProjectPoints

    # Of form {region : region_column}
    regions = {'Rhode Island': 'state'}  # or
    regions = {'Providence': 'county', 'Kent': 'county'}

    res_file = os.path.join(TESTDATADIR, 'nsrdb/', 'ri_100_nsrdb_2012.h5')
    sam_file = os.path.join(TESTDATADIR, 'SAM/wind_gen_standard_losses_0.json')

    pp = ProjectPoints.regions(regions, res_file, sam_file)
    display(pp.df)

        gid                                             config
    0    13  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    1    14  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    2    18  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    3    19  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    4    29  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    5    32  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    6    33  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    7    38  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    8    40  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    9    48  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    10   49  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    11   52  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    12   53  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    13   55  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    14   67  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    15   69  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    16   71  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    17   77  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    18   78  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    19   82  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    20   83  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    21   94  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    22   96  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    23   17  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    24   25  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    25   26  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    26   36  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    27   44  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    28   59  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    29   68  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    30   87  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    31   90  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...
    32   98  /Users/mrossol/Git_Repos/reV/tests/data/SAM/wi...


Command Line Interface (CLI)
----------------------------

Options 3 and 4 above can be run from the Command Line using the
`reV-project-points <https://nrel.github.io/reV/reV/reV.config.cli_project_points.html#rev-project-points>`_
CLI

.. code-block:: bash

    out_file='./project_points.csv'

    TESTDATADIR=reV/tests/data
    res_file=${TESTDATADIR}/nsrdb/ri_100_nsrdb_2012.h5
    sam_file=${TESTDATADIR}/SAM/wind_gen_standard_losses_0.json

    reV-project-points --fpath=${out_file} --res_file=${res_file} --sam_file=${sam_file} from-lat-lons --lat_lon_coords 41.77 -71.74

.. code-block:: bash

    out_file='./project_points.csv'

    TESTDATADIR=../tests/data
    res_file=${TESTDATADIR}/nsrdb/ri_100_nsrdb_2012.h5
    sam_file=${TESTDATADIR}/SAM/wind_gen_standard_losses_0.json

    reV-project-points --fpath=${out_file} --res_file=${res_file} --sam_file=${sam_file} from-regions --region="Rhode Island" --region_col=state
