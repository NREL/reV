# -*- coding: utf-8 -*-
"""Slotted memory framework classes."""


class SlottedDict:
    """Slotted memory dictionary emulator."""

    # make attribute slots for all dictionary keys
    __slots__ = ['var_list']

    def __init__(self):
        self.var_list = []

    def __setitem__(self, key, value):
        """Send data to a slot. Raise KeyError if key is not recognized"""
        if key in self.__slots__:
            if key not in self.var_list:
                self.var_list.append(key)
            setattr(self, key, value)
        else:
            raise KeyError('Could not save "{}" to slotted dictionary. '
                           'The following output variable slots are '
                           'available: {}'.format(key, self.__slots__))

    def __getitem__(self, key):
        """Retrieve data from slot. Raise KeyError if key is not recognized"""
        if key in self.var_list:
            return getattr(self, key)
        else:
            raise KeyError('Variable "{}" has not been saved to this slotted '
                           'dictionary instance. Saved variables are: {}'
                           .format(key, self.keys()))

    def update(self, slotted_dict):
        """Add output variables from another instance into this instance.

        Parameters
        ----------
        slotted_dict : SlottedDict
            An different instance of this class (slotted dictionary class) to
            merge into this instance. Variable data in this instance could be
            overwritten by the new data.
        """

        attrs = slotted_dict.var_list
        for attr in attrs:
            if attr in self.__slots__:
                value = getattr(slotted_dict, attr, None)
                if value is not None:
                    self[attr] = value

    def items(self):
        """Get an items iterator similar to a dictionary.

        Parameters
        ----------
        items : iterator
            [key, value] iterator similar to the output of dict.items()
        """

        keys = self.keys()
        values = self.values()
        return zip(keys, values)

    def keys(self):
        """Get a keys list similar to a dictionary.

        Parameters
        ----------
        key : list
            List of slotted variable names that have been set.
        """
        return [k for k in self.var_list]

    def values(self):
        """Get a values list similar to a dictionary.

        Parameters
        ----------
        values : list
            List of slotted variable values that have been set.
        """
        return [self[k] for k in self.var_list]
