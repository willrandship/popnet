from popnet import *


#training data
train_data = {'target':[],'in':[]}
with open('mnist_train_small.csv') as f:
 for line in f:
  data = line.split(',')

  target = 1 << mpz(data[0])
  px = [int(x) > 127 for x in data] #threshold for now, will do gray coding later

  #append the unary-converted digit target as a desired output
  train_data['target'].append(target)

  #append the generated mpz int as an input
  train_data['in'].append(pack_inputs(px))

test_data = {'target':[],'in':[]}
with open('mnist_test.csv') as f:
 for line in f:
  data = line.split(',')

  target = 1 << mpz(data[0])
  px = [int(x) > 127 for x in data] #threshold for now, will do gray coding later

  #append the unary-converted digit target as a desired output
  test_data['target'].append(target)

  #append the generated mpz int as an input
  test_data['in'].append(pack_inputs(px))

print(train_data['in'][0][0])
print(train_data['target'][0])
print(test_data['in'][0][0])
print(test_data['target'][0])


print(len(train_data['target']))
print(len(test_data['target']))

#model architecture - 3 layers deep, 1 hidden (for now)
#input - 784 binary b/w pixels
#hidden - 2 layers of 1024 nodes convolutionally connected
#output - 10-way selector

#parameter count: 784*1024 + 1024*10 = 813056 edges = 1,626,112 mask/inv bits
# 10 + 1024 = 1,034 thresholds

h1 = Layer(784,100,'h1')
#h2 = Layer(100,50,'h2')
#h3 = Layer(50,25,'h3')
o = Layer(100,10,'o')

from gmpy2 import random_state,mpz_random
seed = random_state(0)

train_count = 0
while 1:
	print('Training for',train_count*1000,'rounds')
	train_count += 1
	for _ in range(1000):
		x = mpz_random(seed,len(train_data['in']))

		#forward pass to populate actuals from training sample
		f1 = h1.forward(train_data['in'][x][0])
		#f2 = h2.forward(f1)
		#f3 = h3.forward(f2)

		#reverse pass to modify parameters
		bp = o.reverse(train_data['target'][x],f1)
		#bp = h3.reverse(bp, f2)
		#bp = h2.reverse(bp, f1)
		bp = h1.reverse(bp,train_data['in'][x][0])

	correct=0
	partial=0
	total=0
	for _ in range(100):
		x = mpz_random(seed,len(train_data['in']))
		#x = mpz_random(seed,len(test_data['in']))

		#print('Testing on '+str(x))

		#state = h1.forward(test_data['in'][x][0])
		state = h1.forward(train_data['in'][x][0])
		#state = h2.forward(state)
		#state = h3.forward(state)
		out = o.forward(state)
		#print(from_gray(out) ,'vs',from_gray(test_data['target'][x]))
		total+=1
		if (out & train_data['target'][x]) > 0:
			partial+=1
		if out==train_data['target'][x]:
			correct+=1
	print(correct,'% Correct,',partial,'% Partially Correct')

