from functools import wraps

class ComputationOrderException(Exception):
    pass

def checker(checked_meth):
    def reset():
        nonlocal computed
        computed = False
        
    def needed(meth):
        @wraps(meth)
        def new_meth(self, *args, **kargs):
            if computed:
                return meth(self, *args, **kargs)
            raise ComputationOrderException(f"{checked_meth.__name__} has to be computed before calling {meth.__name__}.") 
        return new_meth
    
    @wraps(checked_meth)
    def wrapper(self, *args, **kwargs):
        nonlocal computed
        computed = True
        return checked_meth(self, *args, **kwargs)
    
    computed = False
    wrapper.reset = reset
    wrapper.needed = needed
    return wrapper

__all__ = ["checker"]
