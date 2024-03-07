from gmpy2 import mpz,random_state,mpz_random,xmpz
from math import floor

seed_zero = random_state(0)

#takes three integers, returns the 3-bit majority of each bit
#used for training
def maj3(a,b,c):
 #any 2 bits set to 1 = 1 out.
 return (a & b) | (b & c) | (a & c)

#takes a map of wrong positions, and updates the mask and inverse to move accordingly
#if mask was on (enable), and it was right, do nothing to mask or inv
#if mask was off (disable), and it was right, turn mask on
#if mask was on and it was wrong, turn off mask
#if mask was off and it was wrong, flip inv
def wrong2maskinv(wrong,mask,inv):
 inv_out = inv ^ (wrong ^ (~mask))
 mask_out = mask ^ (wrong & mask)
 mask_out = mask_out ^ ((~wrong) & (~mask))
 return mask_out, inv_out

#convert to/from gray coding (symmetric)
def to_gray(x):
	return x ^ (x>>mpz(1))

def from_gray(x):
	t = x.bit_length()
	c = mpz(1)
	while c < t:
		c <<= 1
	while c > 0:
		x ^= x >> c
		c >>= 1
	return x

#process one node forward, returning a bit
def forward(state,mask,inv,threshold):
	return mpz(mpz((state & mask)^inv).bit_count() > threshold)

def forward_nothresh(state,mask,inv):
	return mpz((state & mask)^inv).bit_count()
def forward_nocount(state,mask,inv):
	return mpz((state & mask)^inv)

#process one layer forward, returning an int
def forward_layer(state, masks, invs, threshold):
	return mpz(sum(
		[forward(state, masks[x], invs[x], threshold) << x for x in range(len(masks))]
	))

# reverse a node, given:
# * Ideal activation states
# * Inputs to forward (state, mask, inv, threshold)
#
# produce:
# * new mask, inv, threshold
# To do this
# * move thresholds 1 count closer to criticality (down if on, up if off)
# * adjust mask/inv as a 4-state system relative to correctness
#   * if it was wrong, move away from agreement by 1
#   * if it was right, move toward agreement by 1
#   * 3 = mask 1,inv 0 = enabled + not inverted = total agreement
#   * 2 = mask 0,inv 0 = not enabled + not inverted
#   * 1 = mask 0,inv 1 = not enabled + inverted
#   * 0 = mask 1,inv 1 = enabled + inverted = total disagreement
# needs to return ideal vectors for incoming state as well.
def reverse(ideal, state, mask, inv, threshold, s_in, seed=seed_zero):
	actual = forward_nothresh(state,mask,inv)
	precount = forward_nocount(state,mask,inv)
	result = mpz(actual>threshold)

	#output was what we wanted
	if result==ideal:
		global_agree = 1
	else:
		global_agree = 0

	#-1 is all 1s, -0 is still just 0
	broadcast_ideal = mpz(-ideal)

	wrong = broadcast_ideal ^ precount

	#create change bits for the mask and inv parameters of wrong outputs
	#by creating changes, we can tune the change amount by repeatedly &ing with more randoms.
	#mask_change = wrong & (mpz_random(seed,mpz(1) << wrong.bit_length()))
	#inv_change = wrong & (mpz_random(seed,mpz(1) << wrong.bit_length()))

	#-1 if actual and ideal agree, 0 if they don't
	#agree = (-1 if (actual>threshold)==ideal else 0)

	mask_out, inv_out = wrong2maskinv(wrong,mask,inv)

	#mask_out = (mask ^ mask_change) % (1 << s_in)
	#inv_out = (inv ^ inv_change) % (1 << s_in)

	return mask_out,inv_out,wrong

#take a list [mpz,mpz,mpz] of a specified size(bits) s_in
#return a list [mpz,mpz...] of length s_in such that
#every bit of a single mpz of the input is broadcast to that nth bit on every output
def rotate_mpz_list(l,s_in):
	out = [mpz(0)]*s_in
	for x in reversed(l):
		for i in range(s_in):
			out[i] = (out[i] << 1) + x%2
			x >>= 1
	return out

#perform reverse over a full layer, rather than a single node.
#still a single state, because each layer takes the same state in
#(fully convolutional)
#def reverse_layer(ideals, state, masks, invs, thresholds,s_in):
def reverse_layer(ideals, state, masks, invs, threshold,s_in):
	masks_out = masks
	invs_out = invs
	ideals_out = mpz(0)
	wrongs = masks_out
	for x in range(len(masks)):
		masks_out[x],invs_out[x],wrongs[x] = reverse(
			ideals >> x,state,masks[x],invs[x],threshold,s_in
		)
		#ideals_out = (ideals_out << 1) + agree

	#take majority wrongness opinion to construct backprop ideal
	wrongs_by_input = rotate_mpz_list(wrongs,s_in)
	for x,w in enumerate(wrongs_by_input):
		#output is:
		#majority wrongness as true/false (right/wrong)
		#XOR with output state
		ideals_out = (ideals_out << 1) + mpz( (w.bit_count() <= (s_in//2)) ^ state >> x)

	#agree if it's wrong less than half the time, per input
	#count totals
	#counts = [0]*s_in
	#for x in wrongs:
	#	idx = 0
	#	while x > 0:
	#		if x%2==1:
	#			counts[idx]+=1
	#		x>>=2
	#convert the list of counts into an mpz of average correctness-es
	#wrong_maj = sum([mpz(x > s_in//2) << i for i,x in enumerate(counts)])
	#wrong_maj = sum([mpz(0) if (x<s_in//2) else mpz(0).bit_set(i) for i,x in enumerate(counts)])

	#disagrees = [x.bit_count() > (s_in//2) for x in rotate_mpz_list(wrongs,s_in)]
	#disagrees = rotate_mpz_list(disagrees,1)[0]
	#actuals = [x.bit_count() > (s_in//2) for x in rotate_mpz_list(precounts)]

	#given the state fed in and the wrongs fed back, calculate the backprop ideals
	#ideals_out = wrong_maj ^ state

	return masks_out,invs_out,ideals_out
	#[mask_out, inv_out, threshold_out] = reverse(ideal, state, mask, inv, threshold)

#take several inputs (must all be ints, castable to mpz) and packs
#them into a single larger integer
#if a size is not specified, it is taken to be 8 bits
#to specify a size, pack a tuple with the second param being size
def pack_inputs(*args):
	if len(args)==1 and type(args[0]) is list:
		args = args[0]

	out = 0
	c = 0

	unpack = []

	for x in args:
		if type(x) is tuple:
			l = x[1]
			x = x[0]
		elif type(x) is bool:
			l = 1
		else:
			l = 8
		#add to unpack array
		unpack.append(l)

		#shift into position and add, after clipping max/min to size provided
		out += mpz( max(min(x, 1<<l ),0) << c)
		c += l

	return out,unpack

def unpack_outputs(packed,unpack):
	#'unpack' is a list of bit sizes to unpack, in order
	out = []

	for l in unpack:
		if l > 1:
			out.append(packed % (1<<l))
		else:
			out.append(packed % 2 > 0)
		packed >>= l

	return out

from gmpy2 import random_state,mpz_random
seed=random_state(0)
class Layer:
	def __init__(self,s_in,s_out,name=''):
		self.s_in = s_in
		self.s_out = s_out
		self.name = name
		self.masks = [mpz_random(seed,2<<s_in) for x in range(s_out)]
		self.invs = [mpz_random(seed,2<<s_in) for x in range(s_out)]
		self.threshold = s_in//2

	def __repr__(self):
		return "<Popnet Layer '"+self.name+"' with "+str(self.width)+" nodes>"

	#run a layer forward and return the output state it generates
	def forward(self,state):
		return forward_layer(state,self.masks,self.invs,self.threshold)

	#run a layer backprop and return the ideal for the previous layer
	def reverse(self,ideal,state):
		self.masks,self.invs,ideal_out = reverse_layer(
			ideal,state,self.masks,self.invs,self.threshold,self.s_in )
		return ideal_out

