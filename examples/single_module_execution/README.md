# reV Single-Module Execution

This set of example files demonstrates how to run a single reV analysis module.

The any module can be executed using the following CLI call:

`rev -c "/scratch/gbuster/rev/module_config.json" {module}`

By default, a `rev_status.json` file will be created in the output directory.
Each node utilized in a job will additionally generate their own status jsons upon completion.
Each node makes its own status json to avoid a parallel write conflict to the single `rev_status.json` file.
To collect all of the individual node-level status jsons, the following code can be executed from an ipython console:

```
from reV.pipeline.status import Status
path = '/scratch/gbuster/rev/'
Status.update(path)
```
