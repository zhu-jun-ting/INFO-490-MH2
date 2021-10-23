# this is a python file for mostly used util functions that can used in projects
#  just import this from github and it is okay


def flat(l): 
	for k in l: 
		if not isinstance(k, (list, tuple)): 
			yield k 
		else: 
			for v in flat(k):
				yield v


def pipe(data=None, args=[], *ops):
	if callable(data):
		args = [data, args]
	if not ops == None:
		for op in ops:
			args.append(op)
	if args == []:
		return data
	else:
		if callable(args[0]):
			func = args[0]
			return pipe(func(data), args[1:])
		elif isinstance(args[0], (list, tuple)):
			func = args[0][0]
			sub_args = args[0][1:]
			return pipe(func(data, *sub_args), args[1:])