"""
Generation
"""
import click
import logging


logger = logging.getLogger(__name__)


def sanitize_str(value, subs=('slice', '=', '(', ')', ' ')):
    """Sanitize characters from string."""
    for s in subs:
        value = value.replace(s, '')
    return value


class SAMFilesType(click.ParamType):
    """SAM config files click input argument type."""
    name = 'samfiles'

    @staticmethod
    def convert(value, param, ctx):
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
                raise TypeError('Cannot recognize SAM files type: {} {} {} {}'
                                .format(value, type(value), param, ctx))
        elif isinstance(value, (dict, list, tuple)):
            return value
        else:
            raise TypeError('Cannot recognize SAM files type: {} {} {} {}'
                            .format(value, type(value), param, ctx))


class ProjectPointsType(click.ParamType):
    """Project points click input argument type."""
    name = 'points'

    @staticmethod
    def convert(value, param, ctx):
        """Convert value to slice or list, or return as string."""
        if isinstance(value, str):
            if 'slice' in value:
                # project points is a slice
                value = sanitize_str(value)
                list0 = value.split(',')
                list0 += [None] * (3 - len(list0))
                numeric = [int(x) if str(x) != 'None' else None for x in list0]
                return slice(*numeric)

            elif (('[' in value and ']' in value) or
                  ('(' in value and ')' in value)):
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
                raise TypeError('Cannot recognize points type: {} {} {} {}'
                                .format(value, type(value), param, ctx))
        elif isinstance(value, slice):
            return value
        else:
            raise TypeError('Cannot recognize points type: {} {} {} {}'
                            .format(value, type(value), param, ctx))


class IntType(click.ParamType):
    """Integer click input argument type."""
    name = 'int'

    @staticmethod
    def convert(value, param, ctx):
        """Convert to int or return as None."""
        if isinstance(value, str):
            if 'None' in value:
                return None
            else:
                return int(value)
        elif isinstance(value, int):
            return value
        else:
            raise TypeError('Cannot recognize int type: {} {} {} {}'
                            .format(value, type(value), param, ctx))


class StrType(click.ParamType):
    """String click input argument type."""
    name = 'str'

    @staticmethod
    def convert(value, param, ctx):
        """Convert to int or return as None."""
        if isinstance(value, str):
            if 'None' in value:
                return None
            else:
                return value
        else:
            raise TypeError('Cannot recognize int type: {} {} {} {}'
                            .format(value, type(value), param, ctx))


class ListType(click.ParamType):
    """Base list click input argument type."""
    name = 'list'

    def convert(self, value, param, ctx):
        """Convert string to list."""
        if isinstance(value, str):
            value = sanitize_str(value, subs=['=', '(', ')', ' ', '[', ']',
                                              '"', "'"])
            if value == 'None':
                return None
            list0 = value.split(',')
            return [self.dtype(x) for x in list0]
        elif isinstance(value, list):
            return value
        elif isinstance(value, type(None)):
            return value
        else:
            raise TypeError('Cannot recognize list type: {} {} {} {}'
                            .format(value, type(value), param, ctx))

    @staticmethod
    def dtype(x):
        """Option to enforce a Homogeneous datatype."""
        return x


class FloatListType(ListType):
    """Homogeneous list of floats click input argument type."""
    name = 'floatlist'

    @staticmethod
    def dtype(x):
        """Enforce a homogeneous float datatype."""
        return float(x)


class IntListType(ListType):
    """Homogeneous list of integers click input argument type."""
    name = 'intlist'

    @staticmethod
    def dtype(x):
        """Enforce a homogeneous integer datatype."""
        return int(x)


class StrListType(ListType):
    """Homogeneous list of strings click input argument type."""
    name = 'strlist'

    @staticmethod
    def dtype(x):
        """Enforce a homogeneous string datatype."""
        return str(x)


INT = IntType()
STR = StrType()
SAMFILES = SAMFilesType()
PROJECTPOINTS = ProjectPointsType()
INTLIST = IntListType()
FLOATLIST = IntListType()
STRLIST = StrListType()
