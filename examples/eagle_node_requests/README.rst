reV Eagle Node Requests
=======================

When running reV on Eagle, it's only necessary to specify the allocation and the walltime.
The partition will be chosen automatically and you will be given access to the node's full memory.
So a default execution control block in the config ``.json`` for the standard partition should look like the following:

.. code-block:: json

	"execution_control": {
		"allocation": "rev",
		"nodes": 5,
		"option": "eagle",
		"walltime": 10.0
		},

A node request with high priority in the bigmem partition should look like the following:

.. code-block:: json

	"execution_control": {
		"allocation": "rev",
		"feature": "--qos=high -p bigmem",
		"nodes": 5,
		"option": "eagle",
		"walltime": 10.0
		},

A node request with high priority in the short partition with a 192 GB node should look like the following:

.. code-block:: json

	"execution_control": {
		"allocation": "rev",
		"feature": "--qos=high",
		"memory": 192,
		"nodes": 5,
		"option": "eagle",
		"walltime": 4.0
		},

Note that the way SLURM does memory allocations, if the memory is requested explicitly
in the config ``.json`` and a larger node is received, the user can only use memory up to the requested memory value.
