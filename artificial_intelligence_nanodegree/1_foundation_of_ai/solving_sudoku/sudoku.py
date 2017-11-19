from utils import *
rows = 'ABCDEFGHI'
cols = '123456789'


def cross(a, b):
    ''' Function to create a cross-grid indexes

    Args:
        a: The index of rows
        b: The index of columns
    Returns:
        The concatenated index.
    '''
    return [ele_a + ele_b for ele_a in a for ele_b in b]


def grid_values(grid):
    '''Function to convert a long string format sudoku values into a
    dictionary with indexes

    Args:
        grid: The string containing the values of sudoku.
    Returns:
        The dictionary form of the sudoku puzzle
    '''
    boxes = cross(rows, cols)
    grid_dict = {boxes[i]: v if v != '.' else '123456789'
                 for i, v in enumerate(grid)}
    return grid_dict


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
                values[e] = values[e].replace(elimination_value, '')
    return values


def only_choice(values):
    """Finalize all values that are the only choice for a unit.

    Go through all the units, and whenever there is a unit with a value
    that only fits in one box, assign the value to this box.

    Input: Sudoku in dictionary form.
    Output: Resulting Sudoku in dictionary form after filling in only choices.
    """
    for units in unitlist:
        for digit in '123456789':
            digit_location = [unit for unit in units if digit in values[unit]]
            if len(digit_location) == 1:
                values[digit_location[0]] = digit
    return values


def reduce_puzzle(values):
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len(
            [box for box in values.keys() if len(values[box]) == 1])

        # Your code here: Use the Eliminate Strategy
        values = eliminate(values)

        # Your code here: Use the Only Choice Strategy
        values = only_choice(values)

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


easy_grid = '..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..'
easy_sudoku_dict = grid_values(easy_grid)
# display(sudoku_dict)
# eliminated_dict = eliminate(sudoku_dict)
# display(eliminated_dict)
# only_choice_dict = only_choice(eliminated_dict)
# display(only_choice_dict)

harder_grid = '4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......'
harder_sudoku_dict = grid_values(harder_grid)
solved_harder_sudoku = search(harder_sudoku_dict)
display(solved_harder_sudoku)
