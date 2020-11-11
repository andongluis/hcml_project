
from os import listdir
from os.path import isfile, join

mypath = "./features/"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

fout = open("merged.csv","a")
# first file:
for line in open(onlyfiles[0]):
    fout.write(line)
# now the rest:    
for num in range(1,len(onlyfiles)):
    f = open(onlyfiles[num])
    f.next() # skip the header
    for line in f:
         fout.write(line)
    f.close() # not really needed
fout.close()