'''
Created on Feb 21, 2019

@author: nope
'''
import threading
import contextlib
try:
    from time import perf_counter  as now
except ImportError:
    from time import time as now
import functools

class LogEntry:
    __slots__ = 'key', 'text', 'count', 'timings', 'parent', 'children'
    def __init__(self, text, parent=None, key=None):
        self.text = text
        self.parent = parent
        self.key = text if key is None else text
        self.count = 0
        self.timings = 0
        self.children = dict()
    def __str__(self):
        return "LogEntry(text='%s', key='%s', count=%i, parent='%s', timings=%f, children_count(%i))" % (
            self.text, self.key, self.count, self.parent.key, self.timings, len(self.children)
        )


class TimeIt():
    def __init__(self):
        #self.logs = dict()
        self._thread_state = {}
        
    @contextlib.contextmanager
    def log(self, text, key=None):
        thread = threading.current_thread()
        tname = "%s (%i)" % (thread.getName(), thread.ident)
        state = self._thread_state.get(tname)
        key = text if key is None else key
        if state is None:
            parent = LogEntry("Thread '%s'" % tname)
            state = [parent]
            self._thread_state[tname] = state
        parent = state[-1]
        log_obj = parent.children.get(key)
        if log_obj is None:
            log_obj = LogEntry(text, parent, key)
            parent.children[key] = log_obj
        state.append(log_obj)
        stime = now()
        yield
        stime = now() - stime
        log_obj.count += 1
        log_obj.timings += stime
        plog_obj = state.pop() 
        # just to be sure
        if plog_obj != log_obj:
            raise Exception("Popped log_obj != own log_obj")
        
    def timeit(self, text=None, key=None):
        def decorator_timeit(func):
            @functools.wraps(func)
            def inner(*args, **kwargs):
                innertext = text
                if text is None and key is None:
                    innertext = func.__name__
                with self.log(innertext, key):
                    return func(*args, **kwargs)
            return inner
        return decorator_timeit
        
    def _print_log_recursive(self, log_objs, depth=0, printer=print):
        log_objs = sorted(log_objs, key=lambda x: x.timings, reverse=True)
        for log_obj in log_objs:
            printer("%12.6f %12.6f %8i %s %s" % (
                log_obj.timings,
                (log_obj.timings / log_obj.count) if log_obj.count else -1,
                log_obj.count,
                "  "*depth,
                log_obj.text
            ))
            self._print_log_recursive(log_obj.children.values(), depth+1)
        
    def print_summary(self, printer=print):
        for tname, logs in self._thread_state.items():
            printer("Thread: %s" % tname)
            printer("   %-12s %-12s %-8s" % ("ALL", "PER CALL", "COUNT"))
            self._print_log_recursive(logs[0].children.values(), printer=printer)
        
            
timeit = TimeIt()


if __name__ == '__main__':
    from time import sleep
    
    def da_thread():
        @timeit.timeit()
        def do_stuff():
            with timeit.log("sleep 0.001 seconds"):
                sleep(0.001)
        
        with timeit.log("Wait for 0.5 seconds"):
            sleep(0.5)
        
        with timeit.log("Wait for 4 x 0.2 seconds and do do_stuff"):
            for i in range(4):
                with timeit.log("Wait for 0.2 seconds"):
                    sleep(0.2)
                do_stuff()
        
    threads = []
    for _ in range(5):
        t = threading.Thread(target=da_thread, name="same name")
        t.start()
        threads.append(t)
        timeit.print_summary()
        
    for t in threads:
        t.join()

    timeit.print_summary()