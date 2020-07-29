Run reV locally
===============

`reV Gen <https://nrel.github.io/reV/reV/reV.generation.generation.html#reV.generation.generation.Gen>`_
and `reV Econ <https://nrel.github.io/reV/reV/reV.econ.econ.html#reV.econ.econ.Econ>`_
can be run locally using resource .h5 files stored locally or available via
`HSDS <https://github.com/nrel/hsds-examples>`_.

reV Gen
-------

reV Generation using `PySAM <https://pysam.readthedocs.io/en/latest/>`_ to
compute technologically specific capcity factor means and profiles. reV Gen
uses ``SAM`` technology terms and input configuration files

windpower
+++++++++

Compute wind capacity factors for a given set of latitude and longitude
coordinates:

.. code-block:: python

    import os
    from reV import TESTDATADIR
    from reV.config.project_points import ProjectPoints
    from reV.generation.generation import Gen

    lat_lons = np.array([[ 41.25, -71.66],
                         [ 41.05, -71.74],
                         [ 41.45, -71.66],
                         [ 41.97, -71.78],
                         [ 41.65, -71.74],
                         [ 41.53, -71.7 ],
                         [ 41.25, -71.7 ],
                         [ 41.05, -71.78],
                         [ 42.01, -71.74],
                         [ 41.45, -71.78]])

    res_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    sam_file = os.path.join(TESTDATADIR,
                             'SAM/wind_gen_standard_losses_0.json')

    pp = ProjectPoints.lat_lon_coords(lat_lons, res_file, sam_file)
    gen = Gen.reV_run(tech='windpower', points=pp, sam_files=sam_file,
                      res_file=res_file, max_workers=1, fout=None,
                      output_request=('cf_mean', 'cf_profile'))
    print(gen.out['cf_profile'])

    [[0.319 0.538 0.287 ... 0.496 0.579 0.486]
     [0.382 0.75  0.474 ... 0.595 0.339 0.601]
     [0.696 0.814 0.724 ... 0.66  0.466 0.677]
     ...
     [0.833 0.833 0.823 ... 0.833 0.833 0.833]
     [0.782 0.833 0.833 ... 0.833 0.833 0.833]
     [0.756 0.801 0.833 ... 0.833 0.833 0.833]]

pvwatts
+++++++

NOTE: ``pvwattsv5`` and ``pvwattsv7`` are both available from reV.

Compute pvcapacity factors for all resource gids in a Rhode Island:

.. code-block:: python

    import os
    from reV import TESTDATADIR
    from reV.config.project_points import ProjectPoints
    from reV.generation.generation import Gen

    regions = {'Rhode Island': 'state'}

    res_file = os.path.join(TESTDATADIR, 'nsrdb/', 'ri_100_nsrdb_2012.h5')
    sam_file = os.path.join(TESTDATADIR, 'SAM/naris_pv_1axis_inv13.json')

    pp = ProjectPoints.regions(regions, res_file, sam_file)
    gen = Gen.reV_run(tech='pvwattsv5', points=pp, sam_files=sam_file,
                      res_file=res_file, max_workers=1, fout=None,
                      output_request=('cf_mean', 'cf_profile'))
    print(gen.out['cf_mean'])

    [0.183 0.166 0.177 0.175 0.167 0.183 0.176 0.175 0.176 0.177]

Command Line Interface (CLI)
----------------------------

`reV-gen <https://nrel.github.io/reV/reV/reV.generation.cli_gen.html#rev-gen>`_
can also be run from the command line and will output the results to an .h5
file that can be read with `rex.resource.Resource <https://nrel.github.io/rex/rex/rex.resource.html#rex.resource.Resource>`_.

.. code-block:: bash

    out_file='./project_points.csv'

    TESTDATADIR=reV/tests/data
    res_file=${TESTDATADIR}/wtk/ri_100_wtk_2012.h5
    sam_file=${TESTDATADIR}/SAM/wind_gen_standard_losses_0.json

    reV-gen --tech=windpower--fpath=${out_file} --res_file=${res_file} --sam_file=${sam_file} from-lat-lons --lat_lon_coords 41.77 -71.74

.. code-block:: bash

    out_file='./project_points.csv'

    TESTDATADIR=../tests/data
    res_file=${TESTDATADIR}/nsrdb/ri_100_nsrdb_2012.h5
    sam_file=${TESTDATADIR}/SAM/naris_pv_1axis_inv13.json

    reV-project-points --fpath=${out_file} --res_file=${res_file} --sam_file=${sam_file} from-regions --region="Rhode Island" --region_col=state
