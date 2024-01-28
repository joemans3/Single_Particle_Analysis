#file storing custom error classes
#use the base class Exception to create custom error classes

class InitializationKeys(Exception):
    '''
    Error class for the initialization keys.
    '''
    pass

class InitializationValues(Exception):
    '''
    Error class for the initialization values.
    '''
    pass