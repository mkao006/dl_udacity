import matplotlib as mpl
import matplotlib.pyplot as plt
from util import constraint, displayBoard
from sympy import *


# Declare any required symbolic variables
raise NotImplementedError(
    "TODO: declare symbolic variables for the constraint generators")

# Define diffRow and diffDiag constraints
raise NotImplementedError(
    "TODO: create the diffRow and diffDiag constraint generators")
diffRow = constraint(...)
diffRow = constraint(...)

# Test diffRow and diffDiag
_x = symbols("x:3")

# generate a diffRow instance for testing
raise NotImplementedError(
    "TODO: use your diffRow constraint to generate a diffRow constraint for _x[0] and _x[1]")
diffRow_test = None

assert(len(diffRow_test.free_symbols) == 2)
assert(diffRow_test.subs({_x[0]: 0, _x[1]: 1}) == True)
assert(diffRow_test.subs({_x[0]: 0, _x[1]: 0}) == False)
# partial assignment is not false
assert(diffRow_test.subs({_x[0]: 0}) != False)
print("Passed all diffRow tests.")

# generate a diffDiag instance for testing
raise NotImplementedError(
    "TODO: use your diffDiag constraint to generate a diffDiag constraint for _x[0] and _x[2]")
diffDiag_test = None

assert(len(diffDiag_test.free_symbols) == 2)
assert(diffDiag_test.subs({_x[0]: 0, _x[2]: 2}) == False)
assert(diffDiag_test.subs({_x[0]: 0, _x[2]: 0}) == True)
# partial assignment is not false
assert(diffDiag_test.subs({_x[0]: 0}) != False)
print("Passed all diffDiag tests.")


class NQueensCSP:
    """CSP representation of the N-queens problem

    Parameters
    ----------
    N : Integer
        The side length of a square chess board to use for the problem, and
        the number of queens that must be placed on the board
    """

    def __init__(self, N):
        raise NotImplementedError(
            "TODO: declare symbolic variables in self._vars in the CSP constructor")
        _vars = None
        _domain = set(range(N))
        self.size = N
        self.variables = _vars
        self.domains = {v: _domain for v in _vars}
        self._constraints = {x: set() for x in _vars}

        # add constraints - for each pair of variables xi and xj, create
        # a diffRow(xi, xj) and a diffDiag(xi, xj) instance, and add them
        # to the self._constraints dictionary keyed to both xi and xj;
        # (i.e., add them to both self._constraints[xi] and self._constraints[xj])
        raise NotImplementedError(
            "TODO: add constraints in self._constraints in the CSP constructor")

    @property
    def constraints(self):
        """Read-only list of constraints -- cannot be used for evaluation """
        constraints = set()
        for _cons in self._constraints.values():
            constraints |= _cons
        return list(constraints)

    def is_complete(self, assignment):
        """An assignment is complete if it is consistent, and all constraints
        are satisfied.

        Hint: Backtracking search checks consistency of each assignment, so checking
        for completeness can be done very efficiently

        Parameters
        ----------
        assignment : dict(sympy.Symbol: Integer)
            An assignment of values to variables that have previously been checked
            for consistency with the CSP constraints
        """
        raise NotImplementedError(
            "TODO: implement the is_complete() method of the CSP")

    def is_consistent(self, var, value, assignment):
        """Check consistency of a proposed variable assignment

        self._constraints[x] returns a set of constraints that involve variable `x`.
        An assignment is consistent unless the assignment it causes a constraint to
        return False (partial assignments are always consistent).

        Parameters
        ----------
        var : sympy.Symbol
            One of the symbolic variables in the CSP

        value : Numeric
            A valid value (i.e., in the domain of) the variable `var` for assignment

        assignment : dict(sympy.Symbol: Integer)
            A dictionary mapping CSP variables to row assignment of each queen

        """
        raise NotImplementedError(
            "TODO: implement the is_consistent() method of the CSP")

    def inference(self, var, value):
        """Perform logical inference based on proposed variable assignment

        Returns an empty dictionary by default; function can be overridden to
        check arc-, path-, or k-consistency; returning None signals "failure".

        Parameters
        ----------
        var : sympy.Symbol
            One of the symbolic variables in the CSP

        value : Integer
            A valid value (i.e., in the domain of) the variable `var` for assignment

        Returns
        -------
        dict(sympy.Symbol: Integer) or None
            A partial set of values mapped to variables in the CSP based on inferred
            constraints from previous mappings, or None to indicate failure
        """
        # TODO (Optional): Implement this function based on AIMA discussion
        return {}

    def show(self, assignment):
        """Display a chessboard with queens drawn in the locations specified by an
        assignment

        Parameters
        ----------
        assignment : dict(sympy.Symbol: Integer)
            A dictionary mapping CSP variables to row assignment of each queen

        """
        locations = [(i, assignment[j]) for i, j in enumerate(self.variables)
                     if assignment.get(j, None) is not None]
        displayBoard(locations, self.size)


def select(csp, assignment):
    """Choose an unassigned variable in a constraint satisfaction problem """
    # TODO (Optional): Implement a more sophisticated selection routine from
    # AIMA
    for var in csp.variables:
        if var not in assignment:
            return var
    return None


def order_values(var, assignment, csp):
    """Select the order of the values in the domain of a variable for checking during search;
    the default is lexicographically.
    """
    # TODO (Optional): Implement a more sophisticated search ordering routine
    # from AIMA
    return csp.domains[var]


def backtracking_search(csp):
    """Helper function used to initiate backtracking search """
    return backtrack({}, csp)


def backtrack(assignment, csp):
    """Perform backtracking search for a valid assignment to a CSP

    Parameters
    ----------
    assignment : dict(sympy.Symbol: Integer)
        An partial set of values mapped to variables in the CSP

    csp : CSP
        A problem encoded as a CSP. Interface should include csp.variables, csp.domains,
        csp.inference(), csp.is_consistent(), and csp.is_complete().

    Returns
    -------
    dict(sympy.Symbol: Integer) or None
        A partial set of values mapped to variables in the CSP, or None to indicate failure
    """
    raise NotImplementedError("TODO: complete the backtrack function")


num_queens = 8
csp = NQueensCSP(num_queens)
var = csp.variables[0]
print("CSP problems have variables, each variable has a domain, and the problem has a list of constraints.")
print("Showing the variables for the N-Queens CSP:")
display(csp.variables)
print("Showing domain for {}:".format(var))
display(csp.domains[var])
print("And showing the constraints for {}:".format(var))
display(csp._constraints[var])

print("Solving N-Queens CSP...")
assn = backtracking_search(csp)
if assn is not None:
    csp.show(assn)
    print("Solution found:\n{!s}".format(assn))
else:
    print("No solution found.")
