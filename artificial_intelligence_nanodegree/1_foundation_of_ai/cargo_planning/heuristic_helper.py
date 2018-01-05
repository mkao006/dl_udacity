import signal
import pandas as pd
from timeit import default_timer as timer
from run_search import PrintableProblem
from run_search import PROBLEMS
from run_search import SEARCHES

TIMEOUT_LIMIT = 600


class TimeoutException(Exception):   # Custom exception class
    pass


def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException


signal.signal(signal.SIGALRM, timeout_handler)


def find_optimal_plan(problem_index):
    # We use breadth_first_search to identify the optimal plan since
    # we know it is guranteed.
    search_function = SEARCHES[0][1]
    node = search_function(PROBLEMS[problem_index - 1][1]())
    return {'optimal_length': len(node.solution()),
            'optimal_plan': ['{}{}'.format(s.name, s.args)
                             for s in node.solution()]}


def run_search(problem, search_function, parameter=None):

    start = timer()
    ip = PrintableProblem(problem)
    if parameter is not None:
        node = search_function(ip, parameter)
    else:
        node = search_function(ip)
    end = timer()

    result = {'Time (s)': '{:10.4f}'.format(end - start),
              'Nodes Expanded': ip.states,
              'Path Length': len(node.solution())}

    return result


def create_result_dataframe(problem_index, search_index):
    searches = [SEARCHES[i - 1] for i in map(int, search_index)]

    optimal_plan = find_optimal_plan(problem_index)
    optimal_length = optimal_plan['optimal_length']
    print('\nOptimal Path Length: {}\n'.format(optimal_length))
    print('Optimal Plan:\n')
    for action in optimal_plan['optimal_plan']:
        print(action)

    search_method_name = [s[0] if s[2] == '' else '{} (param={})'.format(s[0], s[2])
                          for s in searches]

    results = list()
    for sname, s, h in searches:
        signal.alarm(TIMEOUT_LIMIT)
        try:
            _p = PROBLEMS[problem_index - 1][1]()
            _h = None if not h else getattr(_p, h)
            results.append(run_search(_p, s, _h))
        except TimeoutException:
            timeout_result = {'Time (s)': '-',
                              'Nodes Expanded': '-',
                              'Path Length': '-'}
            results.append(timeout_result)
        else:
            signal.alarm(0)
    results_df = pd.DataFrame(results, index=search_method_name)
    results_df['optimal'] = [pl == optimal_length
                             for pl in results_df['Path Length']]
    return results_df
