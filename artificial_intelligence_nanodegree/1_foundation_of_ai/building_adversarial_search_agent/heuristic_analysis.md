# Heuristic Analysis - Board Position Scoring

Three heuristic evaluation function was implemented based on the
`Improved Player` which computes the difference in the legal move
available to both players.

One shortcoming of the `Improved Player` is that it does not take into
account of the board positions. If we can incorporate this
information, then the resulting heuristic should be superior to the
`Improved Player`.

Based on this observation, we have come up with three adjustment
factors to the improved player based on three different methods to
score the board position.

The new heuristics all have the following form:

$$
V_c = V_{IP} + \delta_i \quad \quad \quad \delta \in [0, 1)
$$

Where $V_c$ is the new customised heuristic function, $V_{IP}$ the
`Improved Player` heuristic and $\delta_i$ is the bonus for occupying
a favourable position. The bonus is intentionally scaled to fall
within the range of 0 and 1. This property ensures the bonus does not
alter the decision of the `Improved Player` when a clear move is
available; yet when a tie does exist, then the agent will take the
position scored favourably by the new bonus.

The three methods implemented for calculating the bonuses are:

* Custom: Reciprocal Euclidean distance from the center. - figure `d)`
* Custom 2: Reciprocal Manhattan distance from the center.  - figure `c)`
* Custom 3: One step legal move assuming the board is empty.  - figure `b)`

The reasoning behind the calculation of `Custom` and `Custom 2` is
because the further away you are from the center, the closer you are
towards the boundary which poses a restriction on the freedom of
movement.

![Board Scoring Heuristics: Shown above are the different methods of
 scoring the position of a board. The bracket corresponds to the
 position then followed by the value composed by each heuristic
 method. The lighter the colour, the higher the value assigned. In
 figure `a)`, we have the uniform scoring where every position is
 scored equally as in the case of `Improved Player`. Following in
 figure `b)` is the one step ahead legal moves available. The number
 of moves is scaled, so the returned value is between 0 and 1. In
 figure `c)` and `d)` we have the reciprocal Manhattan and Euclidean
 distance respectively.](board_position_score.png "Board Scoring
 Heuristics")

When we compare the legal move scoring as opposed to the distance
based method, we can see that it is indifferent in the centre as they
all have maximum freedom. However, the centre position (3, 3) is
clearly preferred over the neighbouring eight positions.

On the other hand, the Manhattan distance is indifferent diagonally,
diagonal positions such as (3, 0) and (2, 1) are valued
identically. However, due to the restriction of the boundary imposed
on (3, 0), it is clear that the position (2, 1) is superior and this
illustrates the shortcoming of Manhattan distance in this particular
scenario.

This leads us to believe that of the three heuristics, the Euclidean
is likely to perform superior to the two alternatives and is supported
by the playing score below.

| Opponent | AB_Custom | AB_Custom_2 | AB_Custom_3|
|---|:---:|:---:|:---:|
AB_Improved | 54 - 46 | 50 - 50 | 52 - 48 |



