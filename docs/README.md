# Sphinx Documentation

The documentation is built with [Sphinx](http://sphinx-doc.org/index.html).
See their documentation for (a lot) more details.

## Making docs for a fresh repo

To generate docs for a fresh repository, recursively copy this full directory
into the new repo.

1. Install the appropriate packages via conda and pip as detailed below.
2. Remove the reV-specific cli files in `/docs/source/_cli/` but keep
   `/_cli/reV.rst` and rename to the new repo.
3. Remove the `/docs/source/misc/` directory.
4. Run a `grep` of "reV" and replace with your new repository name
5. Update the `conf.py`, `index.rst`, and `api.rst` files with details for the
   new repo.
6. Copy the `/reV/.github/workflows/gh_pages.yml` file to automate doc updates
   (optional).
7. Commit new files to the main/master branch of the new repo.
8. Run the `make html` and `make github` commands below (from the `/docs/`
   folder)
9. Update this readme with anything that's wrong or out of date :)

## Install Requirements

To generate the docs yourself, you'll need the appropriate package:

```
conda install sphinx
conda install sphinx_rtd_theme

pip install ghp-import
pip install sphinx-click
```

## Add any new CLI docs

- Create `cli.rst` and `{repo_name}.rst` files in `source/_cli`
- Add the following to the top of the new `{repo_name}.rst` file:
```
.. click:: module_path:main
   :prog: CLI-Alias # e.g. NSRDB
   :show-nested:
```
- `git commit` and `git push` changes to the main/master branch.
- Make the documentation per below.

## Building HTML Docs

If you're on Mac/Linux (recommended) run: `make html`

If you're on Windows run: `make.bat html`

## Building PDF Docs

To build a PDF, you'll need a latex distribution for your system.

If you're on Mac/Linux (recommended) run: `make latexpdf`

If you're on Windows run: `make.bat latexpdf`

## Pushing to GitHub Pages (gh-pages)

From the main/master branch:

If you're on Mac/Linux (recommended) run: `make github`

If you're on Windows run: `make.bat github`

This `make github` command runs the following (which you could do manually):

```
git branch -D gh-pages
git push origin --delete gh-pages
ghp-import -n -b gh-pages -m "Update documentation" ./_build/html
git checkout gh-pages
git push --set-upstream origin gh-pages
git checkout ${BRANCH}
```
