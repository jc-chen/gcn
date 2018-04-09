import os, random
import sys
from shutil import copy2


m = sys.argv[1]
dest = "/home/jean/Documents/shuffled/" + m + "/"

for i in range(int(m)):
	src = "/home/jean/Documents/cleaned/" + random.choice(os.listdir("/home/jean/Documents/cleaned/"))
	print(i,src)
	copy2(src,dest)

