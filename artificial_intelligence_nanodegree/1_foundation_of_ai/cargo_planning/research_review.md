# Shakey: The Mobile Robot And Its Legacy

Author: Michael C. J. Kao

Shakey was the first general-purpose mobile robot able to reason and construct plans to reach the predetermined goal. Despite its dish-washer appearance and *shakey* movements, the revolution in the software left a profound impact on the field of artificial intelligence.


One of the most significant development resulted from the research was the STanford Research Institute Problem Solver or **STRIPS**. STRIPES is a problem solver which can identify a set of applicable operators which transform the initial world into one that satisfies some state goal condition. The differentiating factor of STRIPES to previous problem solver is the representation of the world in well-formed formulas (wff) of the first-order predicate calculus. This representation later formed part of the `classical` language of AI planning and laid the foundation to action language.


Two notable development phase has taken place since, namely the **Action Description Language** (ADL) and **Planning Domain Definition Language** (PDDL). ADL relaxed several constraints such as handling missing literarals and quantifications to encode more realistic problems. While the PDDL attempts to unify the standards in AI planning in order to improve transparency and consistentcy in order to foster collaboration and progress in the field.

Another significant development stemming from this project is the widespread **A\* search** also utilised in this project. The A* search is the first general search algorithm to use a heuristics functions to estimate the remaining cost to the goal for expansion. This innovative extension has lead to a whole realm of heuristic search algorithm which underlies many of modern application such as pathfinding in Google maps. The power of heuristic search was also demonstrated in this project, when a good heuristic is chosen, it is superior than most non-heuristic method; further, it searches the space more efficiently to return a solution before the time constraints.




Reference:

1. [Nils J. Nilsson (1984). SHAKEY THE ROBOT Technical Note 323](http://www.cs.uml.edu/~holly/91.549/readings/629.pdf)
2. [Richard E. Fikes, Nils J. Nilsson (1971). STRIPS: A New Approach to the Application of Theorem Proving to Problem Solving](http://ai.stanford.edu/~nilsson/OnlinePubs-Nils/PublishedPapers/strips.pdf)
3. [Peter E. Hart, Nils J. Nilsson, Bertram Raphael (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Path](http://ai.stanford.edu/~nilsson/OnlinePubs-Nils/PublishedPapers/astar.pdf)
