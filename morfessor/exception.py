class MorfessorException(Exception):
    """Base class for exceptions in this module."""
    pass


class ArgumentException(Exception):
    pass


class InvalidCategoryError(MorfessorException):
    def __init__(self, category):
        super(InvalidCategoryError, self).__init__(
            self,
            u'This model does not recognize the category {}'.format(
                category))


class InvalidOperationError(MorfessorException):
    def __init__(self, operation, function_name):
        super(InvalidOperationError, self).__init__(
            self,
            (u'This model does not have a method ' +
             u'{}, and therefore cannot perform operation "{}"'.format(
                function_name, operation)))


class UnsupportedConfigurationError(MorfessorException):
    def __init__(self, reason):
        super(UnsupportedConfigurationError, self).__init__(
            self,
            u'This operation is not supported in this program configuration. '
            u'Reason: {}.'.format(reason))
