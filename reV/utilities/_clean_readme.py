import sys


PYPI_DISALLOWED_RST = {"raw::", "<p", "</p", "<img", "---------"}
REMOVE_TEXT = ["A visual summary of this process is given below:", "&nbsp;"]


def _clean(fp):
    """Prep the README file for PyPi distribution"""
    readme = []
    with open(fp, encoding="utf-8") as f:
        for line in f.readlines():
            if any(substr in line for substr in PYPI_DISALLOWED_RST):
                continue
            readme.append(line)

    readme = "".join(readme)
    for substr in REMOVE_TEXT:
        readme = readme.replace(substr, "")

    readme = readme.lstrip()

    with open(fp, "w", encoding="utf-8") as f:
        f.write(readme)


if __name__ == "__main__":
    _clean(*sys.argv[1:])
