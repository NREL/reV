# -*- coding: utf-8 -*-
"""
Custom dtypes for Click.
"""
import click
import logging

from rex.utilities.cli_dtypes import sanitize_str, StrListType

logger = logging.getLogger(__name__)


class SAMFilesType(click.ParamType):
    """SAM config files click input argument type."""
    name = 'samfiles'

    def convert(self, value, param, ctx):
        """Convert value to dict, list, or str."""
        if isinstance(value, str):
            if '{' in value and '}' in value:
                out = {}
                value = sanitize_str(value, subs=('=', '(', ')', '{', '}',
                                                  ' ', "'", '"'))
                key_val_list = value.split(',')
                for key_val in key_val_list:
                    key, val = key_val.split(':')
                    out[str(key)] = str(val)
                return out
            elif '[' in value and ']' in value:
                str_list = StrListType()
                return str_list.convert(value, None, None)
            elif '.json' in value:
                return value
            else:
                self.fail('Cannot recognize SAM files type: {} {}'
                          .format(value, type(value)), param, ctx)
        elif isinstance(value, (dict, list, tuple)):
            return value
        else:
            self.fail('Cannot recognize SAM files type: {} {}'
                      .format(value, type(value)), param, ctx)


class ProjectPointsType(click.ParamType):
    """Project points click input argument type."""
    name = 'points'

    def convert(self, value, param, ctx):
        """Convert value to slice or list, or return as string."""
        if isinstance(value, str):
            if 'slice' in value:
                # project points is a slice
                value = sanitize_str(value)
                list0 = value.split(',')
                list0 += [None] * (3 - len(list0))
                numeric = [int(x) if str(x) != 'None' else None for x in list0]
                return slice(*numeric)

            elif (('[' in value and ']' in value)
                  or ('(' in value and ')' in value)):
                # project points is a list or tuple.
                value = sanitize_str(value, subs=['=', '(', ')', ' ', '[', ']',
                                                  '"', "'"])
                if value == 'None':
                    return None
                list0 = value.split(',')
                return [int(x) for x in list0]

            elif '.csv' in value:
                # project points is a csv file
                return value

            else:
                self.fail('Cannot recognize points type: {} {}'
                          .format(value, type(value)), param, ctx)
        elif isinstance(value, slice):
            return value
        else:
            self.fail('Cannot recognize points type: {} {}'
                      .format(value, type(value)), param, ctx)


SAMFILES = SAMFilesType()
PROJECTPOINTS = ProjectPointsType()
