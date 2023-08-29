Run reV with Losses
===================

`reV Generation <https://nrel.github.io/reV/_autosummary/reV.generation.generation.Gen.html#reV.generation.generation.Gen>`_
can include power curve losses and stochastically scheduled outages.

Power Curve Losses (Wind only)
------------------------------

Instead of simple haircut losses, we can add power curve losses.
The example transformation we will use is a horizontal power curve translation.
To do so, we must specify the ``target_losses_percent`` as well as the name
of the ``transformation``. We specify both of these options with the
``'reV_power_curve_losses'`` key in the SAM config.

.. code-block:: python

    import os
    import json
    import tempfile

    import numpy as np
    import matplotlib.pyplot as plt

    from reV import TESTDATADIR
    from reV.config.project_points import ProjectPoints
    from reV.generation.generation import Gen

    lat_lons = np.array([[ 41.97, -71.78],
                         [ 41.05, -71.74],
                         [ 41.25, -71.66]])

    res_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    sam_file = os.path.join(TESTDATADIR,
                                'SAM/wind_gen_standard_losses_0.json')

    with open(sam_file, 'w+', encoding='utf-8') as fh:
        sam_config = json.load(fh)


    power_curve_loss_info = {
        'target_losses_percent': 40,
        'transformation': 'horizontal_translation'

    }
    with tempfile.TemporaryDirectory() as td:
        sam_fp = os.path.join(td, 'gen.json')
        sam_config.pop('turb_generic_loss', None)
        sam_config['reV_power_curve_losses'] = power_curve_loss_info
        with open(sam_fp, 'w+') as fh:
            fh.write(json.dumps(sam_config))

        output_request=('cf_mean', 'cf_profile', 'gen_profile', 'windspeed')
        pp = ProjectPoints.lat_lon_coords(lat_lons, res_file, sam_fp)
        gen = Gen('windpower' pp, sam_fp, res_file, output_request=output_request)
        gen.run(max_workers=1)
    print(gen.out['cf_profile'])

    [[0.133, 0.202, 0.15 ],
     [0.184, 0.045, 0.242],
     [0.508, 0.119, 0.319],
     ...,
     [0.99 , 1.   , 1.   ],
     [0.688, 1.   , 1.   ],
     [0.628, 1.   , 1.   ]]


Power Curve Losses for a single site
------------------------------------
The reV losses module can be used to compute the power curve shift required to meet
a target loss value for a single input site. To do this, the user must specify the
resource at the site as well as the input power curve and target loss info. An
example of this process is given below

.. code-block:: python

    import os
    import matplotlib.pyplot as plt
    from rex import Resource

    from reV.losses.power_curve import (
        PowerCurve,
        PowerCurveLossesInput,
        PowerCurveWindResource,
        adjust_power_curve,
    )

    site_ind = 100
    res_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    with Resource(res_file) as res:
        temperatures = res["temperature_100m"][:, site_ind]
        pressures = res["pressure_100m"][:, site_ind]
        wind_speeds = res["windspeed_100m"][:, site_ind]

    sam_file = os.path.join(TESTDATADIR, 'SAM/wind_gen_standard_losses_0.json')
    with open(sam_file, 'w+', encoding='utf-8') as fh:
        sam_config = json.load(fh)
    pc_wind_speed = sam_config['wind_turbine_powercurve_windspeeds']
    pc_generation = sam_config['wind_turbine_powercurve_powerout']

    power_curve_loss_info = {
        'target_losses_percent': 5,
        'transformation': 'exponential_stretching'
    }

    power_curve = PowerCurve(pc_wind_speed, pc_generation)
    resource_data = PowerCurveWindResource(temperatures, pressures, wind_speeds)
    target_losses = PowerCurveLossesInput(power_curve_loss_info)

    new_curve = adjust_power_curve(
        power_curve, resource_data, target_losses
    )

    _ = plt.plot(power_curve.wind_speed, power_curve, label='Original')
    _ = plt.plot(new_curve.wind_speed, new_curve, label='5% Losses')
    _ = plt.legend(loc='upper left')
    _ = plt.xlabel("Wind Speed (m/s)")
    _ = plt.ylabel("Generated Power (kW)")
    _ = plt.show()



Outage Losses (Wind and Solar)
------------------------------

We can also tell ``reV`` to stochastically schedule outages based on some
outage information that we pass in. Specifically, we need to provide the
outage ``duration``, the number of outages (``count``), the ``allowed_months``,
as well as the ``percentage_of_capacity_lost`` for each outage.

.. code-block:: python

    import os
    import json
    import tempfile

    import numpy as np
    import matplotlib.pyplot as plt

    from reV import TESTDATADIR
    from reV.config.project_points import ProjectPoints
    from reV.generation.generation import Gen

    lat_lons = np.array([[ 41.05, -71.74],
                         [ 41.25, -71.66]])

    res_file = os.path.join(TESTDATADIR, 'wtk/ri_100_wtk_2012.h5')
    sam_file = os.path.join(TESTDATADIR,
                                'SAM/wind_gen_standard_losses_0.json')

    with open(sam_file, 'r', encoding='utf-8') as fh:
        sam_config = json.load(fh)
    sam_config.pop('wind_farm_losses_percent', None)
    sam_config.pop('turb_generic_loss', None)

    outage_info = [
        {
            'count': 5,
            'duration': 24,
            'percentage_of_capacity_lost': 100,
            'allowed_months': ['January'],
        }
    ]
    with tempfile.TemporaryDirectory() as td:
        sam_fp = os.path.join(td, 'gen.json')
        sam_config['reV_outages'] = outage_info
        with open(sam_fp, 'w+') as fh:
            fh.write(json.dumps(sam_config))

        output_request=('cf_mean', 'cf_profile', 'gen_profile')
        pp = ProjectPoints.lat_lon_coords(lat_lons, res_file, sam_fp)
        gen = Gen('windpower', pp, sam_fp, res_file, output_request=output_request)
        gen.run(max_workers=1)
    print(gen.out['cf_profile'][:744].mean(axis=0))

    [0.67402536, 0.6644584]

