
Forward: 
	bitset( target, (((input vector & mask) ^ invert).popcount() < threshold) )
	keep post-invert vec, threshold - popcount for backward prop
Backward:
	Result ^ desired = badness
	badness = 0 implies less change
	badness = 1 implies more change
	backprop does:
		if bad, threshold was on wrong side (determine correct direction)
			forward_out ^ bad = ideal output (1 or 0)
			move threshold 1 step closer to the critical point(threshold - popcount)
				(-1 if above, +1 if below) regardless of correctness (we want nodes participating!)
				Rapid movement is unnecessary. If it equilibriates, it will get there in a very
				short time regardless. If it doesn't, who cares where we move it?

			post-invert and pre-invert are two vectors
			adjustable flags: invert/mask (for XOR/AND, respectively)
			if an edge is right, we want its input to be more strongly favored
			if an edge is wrong, we want its input to be more strongly opposed
			if it's wishy washy, we want it off.
			so, we can have a 4-state stepping
			favor-disable-disable-disfavor
			with
			enable-pass -> disable-pass -> disable-invert -> enable-invert

	backprop labels inputs from other nodes as good or bad as well.
	nodes will have many good/bad labels.
	disabled (masked) edges do not provide badness, no matter how bad they are.
	enabled edges do.
	badness is counted (by votes) and (magic) determines the final badness score.
		ideally no more than 50%? I'm not really sure where the optimum is there...
	some randomness element reducing movement
		calculate changes (to be applied as XOR), but before applying, AND-mask with
		random 64 bit integer? Should be an explicit reduction in effect.
		For even less change, multiple random masks can be applied.

Needs:

* functions that take vectors of uint64 in, can perform &, ^, popcount, etc over the whole vector,
returning the appropriate data (eg another vector, a uint64)
* functions that can pack/unpack data into/out of the vectors

138 seconds for 100*100e6*3 ops
initial testing shows ~200 million u64s/second is feasible on SIMD-CPU, single core
(i5-3350P with STL templates, forward pass operations in order, and/xor/count in place)
this works out to ~14 GBOPs single core on commodity hardware (after multiplying by 64 parameters
per uint).

For models with billion-parameter counts, this makes them accessible for real time operation.

