#copyright: yueshi@usc.edu
import pandas as pd 
import json
from multiprocessing import Pool
import os

def processFile(inputfile,outputfile):
	'''
	read the json file and parse the file id and case id info and save it 
	'''
	case_ids = set()
	data_arr = []
	for fname in inputfile:
		with open(fname) as data_file:    
			data = json.load(data_file)

		for each_record in data:
			# print (each_record)
			file_id = each_record['file_id']
			case_id =  each_record['cases'][0]['case_id']
			#if case_id in case_ids:
			#	case_ids.add(case_id)

			#else:
		
			data_arr.append([file_id,case_id])

	df = pd.DataFrame(data_arr, columns = ['file_id','case_id'])
	
	df.to_csv(outputfile,index=False)
	

if __name__ == '__main__':


	# modify the input file path when use it.
	inputFile=[]
	data_dir = "../data/"
	for fname in os.listdir(data_dir):
		if fname.find('.json') != -1:
			inputFile.append(data_dir+fname)

	#inputFile = data_dir + "files.2018-11-07.json"
	#outputFile =  data_dir + 'file_case_id.csv'
	outputFile =  data_dir + 'file_case_id_miRNA.csv'
	processFile(inputFile, outputFile)



 




