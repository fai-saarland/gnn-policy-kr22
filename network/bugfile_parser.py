#! /usr/bin/env python

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('bugfile', help='the bugfile to parse')
    parser.add_argument('-c', '--clean', action='store_true',
                        help='write a shorter but equivalent version of the provided bugfile (ordering can change)')
    return parser.parse_args()


class BugInfo:
    def __init__(self, state_vals, bug_value, cost_bound, in_pool):
        self.state_vals = state_vals
        """state as list of integers (values of FDR variables)"""
        self.bug_value = bug_value
        """
        lower bound for how buggy a state is as an integer (bug_value == -1 or bug_value > 0)
        * special value -1 means +infinity, i.e., policy fails to solve solvable state
        * otherwise the policy solved the state, but there is another plan with at least bug_value less cost
        """
        assert bug_value == -1 or bug_value > 0
        self.cost_bound = cost_bound
        """finite integer upper bound for h*(state)"""
        assert 0 <= cost_bound
        self.in_pool = in_pool
        """flag indicating whether represented state is in pool (and has been tested)"""

    def __iter__(self):
        return iter((self.state_vals, self.bug_value, self.cost_bound, self.in_pool))

    def __repr__(self):
        return (f'<BugInfo state_vals:{self.state_vals} bug_value:{self.bug_value} '
                f'cost_bound:{self.cost_bound} in_pool:{self.in_pool}>')

    def get_policy_cost(self):
        """get the cost of the plan the policy found or -1 if no plan was found"""
        return -1 if self.bug_value == -1 else self.cost_bound + self.bug_value


def parse_bug_file(path):
    """parses the provided bugfile and returns a list of BugInfos and (if included in bugfile) the task in sas format"""
    state_map = dict()
    result_map = dict()
    pool = set()
    bugs = []
    sas = ""
    with open(path, 'r') as f:
        def next_line():
            return f.readline().rstrip()
        while line := next_line():
            if line == "begin_sas":
                while (line := f.readline()) != "end_sas\n":
                    sas += line
                continue
            state_id = int(line)
            case = next_line()
            if case == 'state':
                state_vals = [int(x) for x in next_line().split()]
                state_map[state_id] = state_vals
            elif case == 'result':
                bug_value = int(next_line())
                cost_bound = int(next_line())
                if state_id in result_map:
                    old_bug_value, old_cost_bound = result_map[state_id]
                    if bug_value == -1 or old_bug_value == -1:
                        # bug_value of -1 indicates +infinity, i.e., policy fails on solvable state
                        bug_value = -1
                    else:
                        bug_value = max(bug_value, old_bug_value)
                        assert bug_value > 0
                    if cost_bound == -1:
                        # cost bound of -1 indicates no cost bound provided (+infinity)
                        cost_bound = old_cost_bound
                    elif old_cost_bound == -1:
                        pass
                    else:
                        cost_bound = min(cost_bound, old_cost_bound)
                result_map[state_id] = (bug_value, cost_bound)
            else:
                assert case == 'pool'
                pool.add(state_id)
        for state_id, state_vals in state_map.items():
            bug_value, cost_bound = result_map[state_id]
            if cost_bound == -1:
                continue
            bugs.append(BugInfo(state_vals, bug_value, cost_bound, state_id in pool))
        return bugs, sas


def dump(outfile, bugs, sas):
    with open(outfile, 'w') as f:
        if sas:
            f.write(f"begin_sas\n{sas}end_sas\n")
        for i, (vals, bug_value, cost_bound, in_pool) in enumerate(bugs):
            f.write(f'{i}\nstate\n{" ".join([str(x) for x in vals])}\n{i}\nresult\n{bug_value}\n{cost_bound}\n')
            if in_pool:
                f.write(f'{i}\npool\n')


if __name__ == "__main__":
    args = parse_arguments()
    if args.clean:
        bugs, sas = parse_bug_file(args.bugfile)
        dump(args.bugfile, bugs, sas)
