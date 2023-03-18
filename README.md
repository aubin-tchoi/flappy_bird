# Flappy bird

RL and non-RL algorithms for solving a game of Flappy Bird.

## Method

A binary tree that computes the outcome of every possible sequence of decisions (jump of fall).

The decision taken is the branch that has the highest number of leaves with a positive outcome.

This structure allows for many optimizations, for instance if the bird crashes at some node of the tree, there is no
need to compute the children of this node as any resulting leave will have a negative outcome.

### RAM/CPU tradeoff

In its current implementation the tree nodes are actually not stored in memory, only the leaves are stored in a array of
booleans (which is very light on memory). Storing the nodes would allow for tree recycling between each successive step
using the `update` method, which significantly reduces the complexity of the method. However, it would require storing
the positions of all the bars in the nodes. Indeed, in the environment the bars move rather than the bird, therefore in
order to append nodes to the tree we would need to replicate this logic. Instead of 2^depth leaves with boolean values,
we would end up with 2^{depth + 1} - 1 nodes, each of them storing a float (vertical velocity), a list of bars and a
boolean.  


## Tricky aspects

- Bars are shuffled (see FlappyBird.observation())
- Inactive bars are kept in the observation
- There are many offsets to take into account precisely
- There are many possible edge cases not discovered yet
- The help message in the environment seems to be wrong (it is the opposite for pos)
- It seems that the bird first goes forward and then goes up or down in the way its position is compared to the position 
of the bars (it is compared to x_left before the movement, and compared to x_right after it)
