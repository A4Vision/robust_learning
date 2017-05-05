import collections


class SingletonByParameters(type):
    _instances = collections.defaultdict(dict)

    def __call__(cls, *args, **kwargs):
        assert len(kwargs) == 0
        if args not in cls._instances[cls]:
            cls._instances[cls][args] = super(SingletonByParameters, cls).__call__(*args)
        return cls._instances[cls][args]

