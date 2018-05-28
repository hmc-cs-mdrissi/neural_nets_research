def isMatch(jprog, cprog):
	jName = jprog[0].replace('_','').lower()
	cName = cprog[0].replace('_','').lower()
	jFile = jprog[1][:jprog[1].index('.')]
	cFile = cprog[1][:cprog[1].index('.')]
	return (jName == cName) and (jFile == cFile)
d = {}
jlist = []
clist = []
for jprog in jlist:
	for cprog in clist:
		if isMatch(jprog,cprog):
			d[jprog[2]] = cprog[2]
ans = [(k,v) for k,v in d.items()]