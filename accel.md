Manipulating 2D bit arrays (each element either 1 or 0) quickly and efficiently on modern hardware

Representations with minimal wasted space, but still non-sparse

List of rows as integers, eg
1 0 1
1 1 0 = [5,3,4]
1 0 0 

List of columns as integers, eg
1 0 1
1 1 0 = [7,2,4]
1 0 0

Is there an easy way to transpose one of these to the other?
Yes.
Recursively mirror as follows:
Divide the block into the 4 standard quadrants
2 1
3 4
If odd, swap around axis, leaving middle alone
If even, swap across midpoint
Recurse four ways on the swapped quadrants,
leaving off the middle if it exists

