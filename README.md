# Flappy bird

Tree-based algorithms for solving a game of Flappy Bird.

## Usage

Use `python main.py --help` to get a list of the arguments to pass to the main script.

### Dependency management

Dependencies are managed using `poetry`: https://python-poetry.org/docs/basic-usage/

## Method

A binary tree that computes the outcome of every possible sequence of decisions (jump of fall).

The decision taken corresponds to the branch that has the highest number of leaves with a positive outcome.

Additionally, a score can be computed on each leaf with a positive outcome to heuristically guide the bird towards
positions that are *less risky*, typically towards the center with a small velocity. This score is governed by the
parameter `heuristic`, three different scores are available.

This tree structure allows for many optimizations, for instance if the bird crashes at some node of the tree, there is
no need to compute the children of this node as any resulting leave will have a negative outcome.

### RAM/CPU tradeoff

In its current implementation the tree nodes are actually not stored in memory, only the leaves are stored in a array of
booleans (which is very light on memory). Storing the nodes would allow for tree recycling between each successive step
using the `update` method, which significantly reduces the complexity of the method. However, it would require storing
the positions of all the bars in the nodes. Indeed, in the environment the bars move rather than the bird, therefore in
order to append nodes to the tree we would need to replicate this logic. Instead of 2^depth leaves with boolean values,
we would end up with 2^{depth + 1} - 1 nodes, each of them storing a float (vertical velocity), a list of bars and a
boolean.

## Tricky aspects

- Bars are shuffled.
- Inactive bars are kept in the observation.
- There are many offsets to take into account precisely.
- There are many possible edge cases not discovered yet.
- The help message in the environment seems to be wrong (it is the opposite for pos).
- It seems that the bird first goes forward and then goes up or down in the way its position is compared to the position 
of the bars (it is compared to x_left before the movement, and compared to x_right after it).
