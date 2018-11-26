# copyright: yueshi@usc.edu
# modified by yongw@usc.edu
import pandas as pd 
import hashlib
import os 
import numpy as np
from utils import logger
def file_as_bytes(file):
    with file:
        return file.read()

def extractMatrix(dirname):
	'''
	return a dataframe of the miRNA matrix, each row is the miRNA counts for a file_id

	'''
	count = 0
	nRNA = 0
	miRNA_data = []
	for idname in os.listdir(dirname):
		# list all the ids 
		if idname.find("-") != -1:
			idpath = dirname +"/" + idname

			# all the files in each id directory
			for filename in os.listdir(idpath):
				# check the miRNA file
				if filename.find("-") != -1:

					filepath = idpath + "/" + filename
					df = pd.read_csv(filepath,sep="\t")
					# columns = ["miRNA_ID", "read_count"]
					if count ==0:
						# get the miRNA_IDs 
						miRNA_IDs = df.miRNA_ID.values.tolist()
					if nRNA != 0 and nRNA != len(df.index):
						print (filename + " mismatch")

					nRNA = len(df.index)

					id_miRNA_read_counts = [idname] + df.read_count.values.tolist()
					miRNA_data.append(id_miRNA_read_counts)


					count +=1
					# print (df)
	columns = ["file_id"] + miRNA_IDs
	df = pd.DataFrame(miRNA_data, columns=columns)
	return df

def extractLabel(inputfile, case_file):
	df = pd.read_csv(inputfile, sep="\t")
	case = pd.read_csv(case_file, sep="\t",usecols=['case_id', 'disease_type', 'primary_site', 'demographic.gender'])
	disease_list = case['primary_site'].unique().tolist()
	disease = pd.DataFrame(disease_list, columns=["primary_site"])
	disease['primary_code'] = range(1,len(disease)+1)
	df['case_id'] = df['cases.0.case_id']
	
	df = pd.merge(df,case, on='case_id', how='left')
	#print (df.shape)
	df = pd.merge(df,disease, on='primary_site', how='left')
	#print (df[['file_id', 'disease_code']])
	df['label'] = df['primary_code']
	df['gender'] = df['demographic.gender']
	df.loc[df['cases.0.samples.0.sample_type'].str.contains("Control"), 'label'] = -1
	df.loc[df['cases.0.samples.0.sample_type'].str.contains("Cell"), 'label'] = -1
	df.loc[df['cases.0.samples.0.sample_type'].str.contains("Normal"), 'label'] = 0
	tumor_count = df.loc[df.label > 0].shape[0]
	normal_count = df.loc[df.label == 0].shape[0]
	logger.info("{} Normal samples, {} Tumor samples ".format(normal_count,tumor_count))
	columns = ['file_id','label','gender']
	return df[columns],disease

if __name__ == '__main__':


	data_dir ="../data/"
	# Input directory and label file. The directory that holds the data. Modify this when use.
	dirname = data_dir + "miRNA"
	label_file = data_dir + "files_meta.tsv"
	case_file = data_dir + "cases_meta.tsv"	
	#output file
	outputfile = data_dir + "miRNA_matrix.csv"
	disease_code = data_dir + "primary_site_code.csv"
	# extract data
	matrix_df = extractMatrix(dirname)
	label_df,disease = extractLabel(label_file, case_file)

	#merge the two based on the file_id
	result = pd.merge(matrix_df, label_df, on='file_id', how="left")

	#save data
	#drop the rows not tumor nor normal
	result = result.drop(result.loc[result.label < 0].index, axis=0)

	disease.to_csv(disease_code, index=False)
	result.to_csv(outputfile, index=False)
	#print (labeldf)

 




