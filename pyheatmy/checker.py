from functools import wraps

class ComputationOrderException(Exception):
    pass

def checker(checked_meth):
    def reset(col):
        col.__dict__['_'+checked_meth.__name__] = False

    def needed(meth):
        @wraps(meth)
        def new_meth(self, *args, **kargs):
            if '_'+checked_meth.__name__ in self.__dict__ and self.__dict__['_'+checked_meth.__name__]:
                return meth(self, *args, **kargs)
            raise ComputationOrderException(f"{checked_meth.__name__} has to be computed before calling {meth.__name__}.") 
        return new_meth
    
    @wraps(checked_meth)
    def wrapper(self, *args, **kwargs):
        self.__dict__['_'+checked_meth.__name__] = True
        return checked_meth(self, *args, **kwargs)
    
    wrapper.reset = reset
    wrapper.needed = needed
    return wrapper

__all__ = ["checker"]
