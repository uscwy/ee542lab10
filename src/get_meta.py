import json
import os
import pandas as pd
import requests

def processFile(inputfile):
    case_ids = set()
    data_arr = []
    for fname in inputfile:
        with open(fname) as data_file:
            data = json.load(data_file)

    for each_record in data:
        # print (each_record)
        file_id = each_record['file_id']
        case_id =  each_record['cases'][0]['case_id']
        data_arr.append([file_id,case_id])

    df = pd.DataFrame(data_arr, columns = ['file_id','case_id'])
    return df


def retrieveFileMeta(file_ids,outputfile):
    fd = open(outputfile,'w')
    cases_endpt = 'https://api.gdc.cancer.gov/files'

    # The 'fields' parameter is passed as a comma-separated string of single names.
    fields = [
        "file_id",
        "file_name",
        "cases.submitter_id",
        "cases.case_id",
        "data_category",
        "data_type",
        "cases.samples.tumor_descriptor",
        "cases.samples.tissue_type",
        "cases.samples.sample_type",
        "cases.samples.submitter_id",
        "cases.samples.sample_id",
        "cases.samples.portions.analytes.aliquots.aliquot_id",
        "cases.samples.portions.analytes.aliquots.submitter_id"
        ]

    filters = {
        "op":"in",
        "content":{
            "field":"files.file_id",
            "value": file_ids.tolist()
        }
    }
    #print(filters)
    fields = ','.join(fields)

    params = {
        "filters" : filters,
        "fields": fields,
        "format": "TSV",
        "pretty": "true",
        "size": file_ids.shape[0]
    }
    # print (params)
    #print (filters)
    #print (fields)
    
    
    response = requests.post(cases_endpt, headers = {"Content-Type": "application/json"},json = params)
    fd.write(response.content.decode("utf-8"))
    fd.close()

    # print(response.content)
def retrieveCaseMeta(file_ids,outputfile):
    '''

    Get the tsv metadata for the list of case_ids
    Args:
        file_ids: numpy array of file_ids
        outputfile: the output filename

    '''

    fd = open(outputfile,'w')
    cases_endpt = 'https://api.gdc.cancer.gov/cases'


    filters = {
        "op":"in",
        "content":{
            "field":"cases.case_id",
            "value": file_ids.tolist()
        }
    }

    # print (filters)
    #expand group is diagnosis and demoragphic
    params = {
        "filters" : filters,
        "expand" : "diagnoses,demographic,exposures",
        "format": "TSV",
        "pretty": "true",
        "size": file_ids.shape[0]
    }
    # print (params)
    #print (filters)
    #print (fields)
    
    
    response = requests.post(cases_endpt, headers = {"Content-Type": "application/json"},json = params)
    # print (response.content.decode("utf-8"))
    fd.write(response.content.decode("utf-8"))
    fd.close()

def genCasePayload(file_ids,payloadfile):
    '''
    Used for the curl method to generate the file payload.
    '''

    fd = open(payloadfile,"w")
    filters = {
        "filters":{
            "op":"in",
            "content":{
                "field":"cases.case_id",
                "value": file_ids.tolist()
            }
        },
        "format":"TSV",
        "expand" : "diagnoses,demographic,exposures",
        "size": "1000",
        "pretty": "true"
    }
    json_str = json.dumps(filters)
    fd.write(json_str)
    fd.close()
    # return json_str

def genFilePayload(file_ids,payloadfile):
    '''
    Used for the curl method to generate the payload.
    '''


    fd = open(payloadfile,"w")
    filters = {
        "filters":{
            "op":"in",
            "content":{
                "field":"files.file_id",
                "value": file_ids.tolist()
            }
        },
        "format":"TSV",
        "fields":"file_id,file_name,cases.submitter_id,cases.case_id,data_category,data_type,cases.samples.tumor_descriptor,cases.samples.tissue_type,cases.samples.sample_type,cases.samples.submitter_id,cases.samples.sample_id,cases.samples.portions.analytes.aliquots.aliquot_id,cases.samples.portions.analytes.aliquots.submitter_id",
        "pretty":"true",
        "size": "1000"
    }
    json_str = json.dumps(filters)
    fd.write(json_str)
    fd.close()



inputFile=[]
data_dir = "../data/"
for fname in os.listdir(data_dir):
    if fname.find('.json') != -1:
        inputFile.append(data_dir+fname)

#inputFile = data_dir + "files.2018-11-07.json"
df_file = processFile(inputFile)
df_file.to_csv(data_dir + 'file_case_id.csv',index=False)
print (df_file.shape) 
file_ids = df_file.file_id.values
case_ids = df_file.case_id.values
# print(case_ids)

fileids_meta_outfile = data_dir + "files_meta.tsv"
caseids_meta_outfile = data_dir + "cases_meta.tsv"
# python request method
retrieveFileMeta(file_ids,fileids_meta_outfile)
retrieveCaseMeta(case_ids,caseids_meta_outfile)

