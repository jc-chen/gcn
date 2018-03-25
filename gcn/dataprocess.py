import os


input_url = '../../tem/'
output_url = '../../tem_processed/'

for file in os.listdir(input_url):
	with open(input_url+file,'r') as fin:
		with open(output_url+file,'w') as fout:
			for line in fin:
				fout.write(line.replace("*^","*10^"))

