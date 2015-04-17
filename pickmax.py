import sys, re
max = 0
config = ""
dev = {}
test = {}
listFile = open(sys.argv[1], 'r')
for l in listFile:
	f = open(l.strip(), 'r')
	overall = f.readlines()[3].strip()
	f.close()
	
	f = open(l.strip(), 'r')
	for line in f:
		line = line.strip()
		m = re.search("MAP", line)
		if(m):
			w = line.split(" ")
			iter = w[3]
			currconfig =  overall + "," + iter
			score = float(w[5])
			m = re.search("Dev", line)
			if(m):
				
				if(score > max):
					max = score
					config = currconfig
				dev[currconfig] = score
			else:
				test[currconfig] = score
	f.close()
print len(dev), len(test)
print config, dev[config], test[config]					
out = open("debug", 'w')
for w in sorted(dev.keys()):
	out.write(w + "," + str(dev[w]) + "," + str(test[w]) + "\n")			
