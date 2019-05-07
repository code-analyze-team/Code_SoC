"""
manage global values
for inter-files variable sharing
"""


class global_value:

    def __init__(self):
        global _global_dict
        _global_dict = {}
        self.global_dict = _global_dict

    def set_value(self, key, value):
        """define a global variable"""
        self.global_dict.setdefault(key, '')
        self.global_dict[key] = value

    def get_value(self, key, default_value):
        """
        return value given a key
        if no value available, return default_value
        """
        try:
            return self.global_dict[key]
        except KeyError:
            return default_value

