import time
assignments = []
rows = 'ABCDEFGHI'
cols = '123456789'


def assign_value(values, box, value):
    """
    Please use this function to update your values dictionary!
    Assigns a value to a given box. If it updates the board record it.
    Args:
        values: a dictionary of the form {'box_name': '123456789', ...}
        box: The key of the value dictionary where the value should go.
        value: The value to be assigned

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """

    # Don't waste memory appending actions that don't actually change any
    # values
    if values[box] == value:
        return values

    values[box] = value
    if len(value) == 1:
        assignments.append(values.copy())
    return values


def cross(a, b):
    ''' Function to create a cross-grid indexes

    Args:
        a: The index of rows
        b: The index of columns
    Returns:
        The concatenated index.
    '''
    return [ele_a + ele_b for ele_a in a for ele_b in b]


boxes = cross(rows, cols)

row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI')
                for cs in ('123', '456', '789')]
diag_units = [[i + j for i, j in zip(list(rows), list(cols))],
              [i + j for i, j in zip(list(rows[::-1]), list(cols))]]
unitlist = row_units + column_units + square_units + diag_units
units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s], [])) - set([s])) for s in boxes)


def grid_values(grid):
    '''Function to convert a long string format sudoku values into a
    dictionary with indexes

    Args:
        grid: The string containing the values of sudoku.
    Returns:
        The dictionary form of the sudoku puzzle
    '''

    grid_dict = {boxes[i]: v if v != '.' else '123456789'
                 for i, v in enumerate(grid)}
    return grid_dict


def display(values):
    """
    Display the values as a 2-D grid.
    Input: The sudoku in dictionary form
    Output: None
    """
    width = 1 + max(len(values[s]) for s in boxes)
    line = '+'.join(['-' * (width * 3)] * 3)
    for r in rows:
        print(''.join(values[r + c].center(width) + ('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF':
            print(line)
    return


def eliminate(values):
    '''Eliminate values from peers of each box with a single value.

    Go through all the boxes, and whenever there is a box with a single value,
    eliminate this value from the set of values of all its peers.

    Args:
        values: Sudoku in dictionary form.
    Returns:
        Resulting Sudoku in dictionary form after eliminating values.
    '''
    solved_keys = [k for k, v in values.items() if len(v) == 1]
    for b in boxes:
        if b in solved_keys:
            elimination_box = peers[b]
            elimination_value = values[b]
            for e in elimination_box:
                new_value = values[e].replace(elimination_value, '')
                assign_value(values, e, new_value)
    return values


def only_choice(values):
    """Finalize all values that are the only choice for a unit.

    Go through all the units, and whenever there is a unit with a value
    that only fits in one box, assign the value to this box.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the only choice assigned from peers.
    """
    for units in unitlist:
        for digit in '123456789':
            digit_location = [unit for unit in units if digit in values[unit]]
            if len(digit_location) == 1:
                assign_value(values, digit_location[0], digit)
    return values


def naked_twins(values):
    """Eliminate values using the naked twins strategy.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the naked twins eliminated from peers.
    """

    # Find all instances of naked twins
    # Eliminate the naked twins as possibilities for their peers

    # Find any box that has only two possible values
    double_key = [k for k, v in values.items() if len(v) == 2]

    # A twin is where the value of the two keys are identical and also
    # they are peers of each other.
    #
    used_i = []
    twins = []
    for i in double_key:
        used_i.append(i)
        for j in double_key:
            if values[i] == values[j] and j in peers[i] and j not in used_i:
                twins.append((i, j))

    for twin_i, twin_j in twins:
        for l in unitlist:
            current_twin_set = set([twin_i, twin_j])
            if current_twin_set.issubset(l):
                eliminate_set = set(l) - current_twin_set
                for es in eliminate_set:
                    if len(values[twin_i]) == 2 and len(values[twin_j]) == 2:
                        new_value = (values[es]
                                     .replace(values[twin_i][0], '')
                                     .replace(values[twin_i][1], ''))
                        assign_value(values, es, new_value)

    return values


def reduce_puzzle(values):
    ''' Reduce the puzzle using constrained propagation.

    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        the values dictionary with the specified strategies applied.

    '''
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len(
            [box for box in values.keys() if len(values[box]) == 1])

        # Your code here: Use the Eliminate Strategy
        values = eliminate(values)

        # Your code here: Use the Only Choice Strategy
        values = only_choice(values)

        # Remove naked twins
        values = naked_twins(values)

        # Check how many boxes have a determined value, to compare
        solved_values_after = len(
            [box for box in values.keys() if len(values[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available
        # values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values


def search(values):
    '''Using depth-first search and propagation, create a search tree and
    solve the sudoku.
    Args:
        values(dict): a dictionary of the form {'box_name': '123456789', ...}

    Returns:
        If the unique solution is reached, then the solution is returned. Otherwise, False.
    '''
    reduced_puzzle = reduce_puzzle(values)

    if reduced_puzzle is False:
        return False  # Failed earlier
    if all(len(reduced_puzzle[s]) == 1 for s in boxes):
        return reduced_puzzle

    n, s = min((len(reduced_puzzle[s]), s)
               for s in boxes if len(reduced_puzzle[s]) > 1)
    # Now use recurrence to solve each one of the resulting sudokus, and
    for value in reduced_puzzle[s]:
        new_sudoku = reduced_puzzle.copy()
        new_sudoku[s] = value
        attempt = search(new_sudoku)
        if attempt:
            return attempt


def solve(grid):
    """
    Find the solution to a Sudoku grid.
    Args:
        grid(string): a string representing a sudoku grid.
            Example: '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    Returns:
        The dictionary representation of the final sudoku grid. False if no solution exists.
    """
    start = time.time()
    solution = search(grid_values(grid))
    time_to_solve = time.time() - start
    print('Time to solve: {:.3f} seconds'.format(time_to_solve))
    return solution


if __name__ == '__main__':
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(solve(diag_sudoku_grid))

    try:
        from visualize import visualize_assignments
        visualize_assignments(assignments)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
