
Assume a fully connected network of layers, for now.
Future work can sparsify this, but for now I just want it to work.

The architecture demands per-node value vs ideal storage
It also demands batched sampling for training (otherwise training never converges)

Each edge has a weight, in the form of invert and mask bits.
mask dictates whether abs(weight) is 0 or 1
invert dictates whether sign(weight) is + or -

Each node has a result output, per forward propagation.
Each node has an ideal, per backpropagation.
Each node has a voting value, storing idealized movement.

Inference is done as follows:

Take the input vector, which has 1 bit per edge.
Bitwise-XOR the input vector with invert vector to produce the signed input.
Bitwise-AND this with the mask vector to produce the population
Count the population (eg popcnt) to produce the count
Count the mask and bit shift right by 1 to produce the threshold.
Compare the count vs the threshold.
Store the result in the appropriate place in the next input vector.

By using count(mask)/2 as the threshold, we observe the following:
* There are a maximum of count(mask) inputs contributing to the vote
* The threshold is maintained at as close to 50% as possible, rounding down.

Backpropagation is done as follows:

Take the result of a forward inference, including all input and intermediate vectors.
Take the ideal final output for the given input (what to train against)
For each node in the output, use the ideal result and the connecting weights to backtrack
ideal results on each node feeding it.
* Each backprop node will have an integer storing sum(votes). The final ideal is the sign of this.
* If the mask for an edge is 0, no change is applied.
* If the invert for an edge is 1, invert the influence direction for the given vote.
* The ideal for the input nodes is simply the input value.

Once this has been determined for all nodes, iterate over each edge and determine the following:
If the input and output ideal are the same, this is a vote for a weight change of +1.
If the input and output ideal are not the same, this is a vote for a weight change of -1.
These votes are 1-bit for a single inference, but are batched together in groups by summing.
This sum is simply popcount(bits) - bits/2, to account for the 0/-1 discrepancy.

Once this vote has been determined for all edges across the batch, it is compared to the learning
rate threshold (LRT). If the vote is clearer than the LRT, then the effect is applied to that edge.
-1 <-> -0 (mask off, inv on)
-0 <-> +0 (mask off, inv off)
+0 <-> +1 (mask on, inv off)
As training progresses, to ensure stability, the LRT should gradually decrease from 0.5 to 0.
For a decrement, the vote must be < total votes * LRT
For an increment, the vote must be > total votes * (1-LRT)

This ends up requiring the following data structures:

edge weights (2 bits per edge per model)
node ideal votesum (1 int per node per sample, rarely if ever >16 bit)
node ideal sign (1 bit per node per sample)
edge movement votes (1 int per node per batch, rarely >16 bit, = log2(batch size))

