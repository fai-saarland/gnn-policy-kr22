from pathlib import Path

def read_plans(path: Path):
    problems = dict()
    directories = ["plans_original", "plans_retrained", "plans_continued"]
    for directory in directories:
        for dpath in path.glob(directory + '/*'):
            if dpath.is_dir():
                problems[dpath] = dict()

                for filename in dpath.glob('*.plan'):
                    problem = filename.stem
                    if problem not in problems[dpath]: problems[dpath][problem] = dict()
                    with filename.open('r') as fd:
                        plan = [ line.strip('\n') for line in fd.readlines() if line[0] == '(' ]
                    problems[dpath][problem]['planner'] = dict(length=len(plan), plan=plan)

                for suffix in [ 'policy']:
                    for filename in dpath.glob(f'*.{suffix}'):
                        problem = filename.stem
                        if problem not in problems[dpath]: problems[dpath][problem] = dict()
                        with filename.open('r') as fd:
                            plan = []
                            reading_plan = False
                            for line in fd.readlines():
                                line = line.strip('\n')
                                if line.find('Found valid plan') > 0:
                                    reading_plan = True
                                    continue
                                if reading_plan:
                                    fields = line.split(' ')
                                    if len(fields) >= 5 and fields[4][-1] == ':' and fields[4][:-1].isdigit():
                                        plan.append(' '.join(fields[5:]))
                            if reading_plan:
                                problems[dpath][problem][suffix] = dict(length=len(plan), plan=plan)
    return problems

def fill_table(table, problems, suffix):
    for dpath in problems:
        domain = dpath.name
        size = len(problems[dpath].keys())
        if domain not in table:
            table[domain] = dict(size=size)

        solved = [ problem for problem in problems[dpath] if problem in problems[dpath] and suffix in problems[dpath][problem] ]
        opt_solved = [ problem for problem in problems[dpath] if problem in problems[dpath] and 'planner' in problems[dpath][problem] ]
        assert len(solved) <= size and len(opt_solved) <= size

        lengths = [ problems[dpath][problem][suffix]['length'] for problem in solved ]
        L = sum(lengths)

        solved_and_opt = set(solved) & set(opt_solved)
        lengths = [ problems[dpath][problem][suffix]['length'] for problem in solved_and_opt ]
        opt_lengths = [ problems[dpath][problem]['planner']['length'] for problem in solved_and_opt ]
        PL, OL, N = sum(lengths), sum(opt_lengths), len(solved_and_opt)

        table[domain][suffix] = dict(solved=len(solved), L=L, PL=PL, OL=OL, N=N)

def get_table_row(name, size, record):
    row = f'{name:>47s} {{:>5s}}'.format(f'({size:,d})')
    for suffix in [ 'policy']:
        solved = record[suffix]['solved']
        assert size > 0, f'{name}'
        percentage = f'({int(100 * solved / size)}\\%)'
        L = record[suffix]['L']
        PL = record[suffix]['PL']
        OL = record[suffix]['OL']
        N = record[suffix]['N']
        if OL > 0:
            PQ = PL / OL
            quality = f'{PQ:>7.4f} = {PL:>6,d} / {OL:>6,d} {{:>5s}}'.format(f'({N})')
        else:
            quality = '{:>31s}'.format('---')
        row += f' & {solved:3d} {percentage:>7s} & {L:>6,d} & {quality:20s}'
    return row

def print_table(table):
    totals = dict(size=0)
    for domain in table:
        size = table[domain]['size']
        totals['size'] += size

        for suffix in [ 'policy']:
            if suffix not in totals: totals[suffix] = dict()
            if 'solved' not in totals[suffix]: totals[suffix]['solved'] = 0
            totals[suffix]['solved'] += table[domain][suffix]['solved']
            if 'L' not in totals[suffix]: totals[suffix]['L'] = 0
            totals[suffix]['L'] += table[domain][suffix]['L']
            if 'PL' not in totals[suffix]: totals[suffix]['PL'] = 0
            totals[suffix]['PL'] += table[domain][suffix]['PL']
            if 'OL' not in totals[suffix]: totals[suffix]['OL'] = 0
            totals[suffix]['OL'] += table[domain][suffix]['OL']
            if 'N' not in totals[suffix]: totals[suffix]['N'] = 0
            totals[suffix]['N'] += table[domain][suffix]['N']

        if size > 0:
            row = get_table_row(domain, size, table[domain])
            print(f'{row} \\\\')
    row = get_table_row('Total', totals['size'], totals)
    print(f'\\midrule\n{row} \\\\')

if __name__ == "__main__":
    # read plans in (domain) folders
    problems = read_plans(Path('.'))

    # create table
    table = dict()
    fill_table(table, problems, 'policy')
    #fill_table(table, problems, 'markovian')

    # print table
    print_table(table)

