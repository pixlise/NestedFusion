from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import datetime
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import json
import pickle

np.random.seed(42)

from formats import experiment_pb2
from formats import  quantification_pb2

from skimage import io
import pandas as pd

import warnings
# warnings.filterwarnings("ignore")

import boto3

base_path = os.path.dirname(__file__)


def reload_data_dir():
    if os.path.isdir(base_path+'/data'):
        shutil.rmtree(base_path+'/data')

    os.mkdir(base_path+'/data')
    s3r = boto3.resource('s3', aws_access_key_id=os.environ['PIXLISE_AWS_KEY_ID'], 
                        aws_secret_access_key=os.environ['PIXLISE_AWS_KEY'])
    buck = s3r.Bucket('prodpixlise-users0a0eea89-3vbmmgfzkohy')


    os.mkdir(base_path+'/data/Quantifications')
    for obj in tqdm(list(buck.objects.filter(Prefix='UserContent/shared'))):
        if len(obj.key.split('/'))>3 and obj.key.split('/')[3] =='Quantifications' and obj.key.split('.')[-1] == 'bin':
            dataset = obj.key.split('/')[2]
            if not os.path.isdir(base_path+'/data/Quantifications/{}'.format(dataset)):
                os.mkdir(base_path+'/data/Quantifications/{}'.format(dataset))
            buck.download_file(obj.key,base_path+"/data/Quantifications/{}/{}".format(dataset,obj.key.split('/')[-1]))
        
        if len(obj.key.split('/'))>3 and obj.key.split('/')[3] =='Quantifications' and obj.key.split('.')[-1] == 'json':
            dataset = obj.key.split('/')[2]
            if not os.path.isdir(base_path+'/data/Quantifications/{}'.format(dataset)):
                os.mkdir(base_path+'/data/Quantifications/{}'.format(dataset))
            buck.download_file(obj.key,base_path+"/data/Quantifications/{}/{}".format(dataset,obj.key.split('/')[-1]))

    os.mkdir(base_path+'/data/ROI')
    for obj in tqdm(list(buck.objects.filter(Prefix='UserContent/shared'))):
        if len(obj.key.split('/'))>3 and obj.key.split('/')[3] =='ROI.json':
            dataset = obj.key.split('/')[2]
            if not os.path.isdir(base_path+'/data/ROI/{}'.format(dataset)):
                os.mkdir(base_path+'/data/ROI/{}'.format(dataset))
            buck.download_file(obj.key,base_path+"/data/ROI/{}/{}".format(dataset,obj.key.split('/')[-1]))
    
    s3r = boto3.resource('s3', aws_access_key_id=os.environ['PIXLISE_AWS_KEY_ID'], 
                        aws_secret_access_key=os.environ['PIXLISE_AWS_KEY'])
    buck = s3r.Bucket('prodpixlise-datasets0030ee04-kaj9yhu41sb5')

    os.mkdir(base_path+'/data/Datasets')
    os.mkdir(base_path+'/data/RGBU')
    for obj in tqdm(list(buck.objects.filter(Prefix='Datasets'))):
        if obj.key.split('/')[-1] =='dataset.bin':
            dataset = obj.key.split('/')[1]
            if not os.path.isdir(base_path+'/data/Datasets/{}'.format(dataset)):
                os.mkdir(base_path+'/data/Datasets/{}'.format(dataset))
            buck.download_file(obj.key,base_path+"/data/Datasets/{}/{}".format(dataset,obj.key.split('/')[-1]))
        if obj.key.split('.')[-1] in ['png','jpg']:
            dataset = obj.key.split('/')[1]
            if not os.path.isdir(base_path+'/data/Datasets/{}'.format(dataset)):
                os.mkdir(base_path+'/data/Datasets/{}'.format(dataset))
            buck.download_file(obj.key,base_path+"/data/Datasets/{}/{}".format(dataset,obj.key.split('/')[-1]))
        if obj.key.split('.')[-1] in ['tif']:
            dataset = obj.key.split('/')[1]
            if not os.path.isdir(base_path+'/data/RGBU/{}'.format(dataset)):
                os.mkdir(base_path+'/data/RGBU/{}'.format(dataset))
            buck.download_file(obj.key,base_path+"/data/RGBU/{}/{}".format(dataset,obj.key.split('/')[-1]))
    
    generate_metadata_df()

def generate_metadata_df():
    df = load_dataset_dataframe()
    df = df.groupby('dataset',as_index=False).first()[['dataset','title','sol','rtt','sclk','site','target','site_id','target_id','context_image']]
    df.to_pickle('data/metadata.pkl')


def load_dataset_dataframe():
    if not os.path.isdir(base_path+'/data'):
            print('Data directory not found! Run with `-r` to reload data directory')
            return
    dataset_paths = [path for path in os.listdir(base_path+"/data/Datasets/")]
    datasets = [experiment_pb2.Experiment() for path in dataset_paths]
    for i in tqdm(range(len(dataset_paths)),leave=False,desc="Loading Protobuf Files"):
      with open(base_path+f'/data/Datasets/{dataset_paths[i]}/dataset.bin', mode='rb') as file:
        datasets[i].ParseFromString(file.read())
    
    dataset_metadata = []
    dataset_idx = 0
    failed_datasets = []
    for dataset, path in tqdm(zip(datasets,dataset_paths),total=len(datasets),leave=False, desc="Processing Datasets"):
      try:
        readtype_index = list(dataset.meta_labels).index('READTYPE')
        detector_id_index = list(dataset.meta_labels).index('DETECTOR_ID')
        pmc_index = list(dataset.meta_labels).index('PMC')
        livetime_index = list(dataset.meta_labels).index('LIVETIME')
      except:
        failed_datasets.append(dataset.title)
        continue

      for loc in dataset.locations:
        if loc.detectors:
            detector = loc.detectors[0]
        else:
            continue
        
        readtype = [i.svalue for i in detector.meta if i.label_idx == readtype_index]
        if readtype:
            readtype = readtype[0]
        else:
            readtype = "N/A"
        
        detector_id = [i.svalue for i in detector.meta if i.label_idx == detector_id_index]
        if detector_id:
            detector_id = detector_id[0]
        else:
            detector_id = "N/A"

        pmc = [i.ivalue for i in detector.meta if i.label_idx == pmc_index]
        if pmc:
            pmc = int(pmc[0])
        else:
            continue
        livetime = [i.fvalue for i in detector.meta if i.label_idx == livetime_index]
        if livetime:
            livetime = float(livetime[0])
        else:
            continue
        if livetime <= 0:
            continue
        
        dataset_metadata.append({
            'context_image' : base_path+f'/data/Datasets/{path}/{dataset.main_context_image}',
            'dataset':path,
            'location_id':loc.id,
            'pmc':pmc,
            'livetime':livetime,
            'image_i': loc.beam.image_i,
            'image_j': loc.beam.image_j,
            'readtype': readtype,
            'sol':dataset.sol,
            'rtt':dataset.rtt,
            'sclk':dataset.sclk,
            'site':dataset.site,
            'target':dataset.target,
            'site_id':dataset.site_id,
            'target_id':dataset.target_id,
            'title':dataset.title
        })
    
    df = pd.DataFrame(dataset_metadata)
    
    # print(f'Successfully processed datasets: \n{df.name.unique()}\n\n')
    # if len(failed_datasets):
    #     print(f'Failed to process datasets: \n{failed_datasets}\n\n')
 
    return df

def check_rgbu(dataset):
    if not os.path.isdir(base_path+'/data'):
            print('Data directory not found! Run with `-r` to reload data directory')
            return     
    has_dir = os.path.isdir(base_path+f'/data/RGBU/{dataset}')
    return has_dir and get_pmc_rgbu_map(dataset)

def rgbu_label_to_index(s):
    return{
        0: 'Near IR',
        1: 'Green',
        2: 'Blue',
        3: 'Ultraviolet'
    }[s] 
    
def rgbu_index_to_label(i):
    return[ 'Near IR',
            'Green',
            'Blue',
            'Ultraviolet'][i]                  
    
def load_rgbu_array(dataset,filename=None):
    if not os.path.isdir(base_path+'/data'):
            print('Data directory not found! Run with `-r` to reload data directory')
            return                  
    if filename:
        if not os.path.isfile(base_path+f'/data/RGBU/{dataset}/{filename}'):
                print(f'RGBU tif not found at {dataset}/{filename}')
                return
        rgbu_array = io.imread(base_path+f'/data/RGBU/{dataset}/{filename}')
    else:
        if not os.path.isdir(base_path+f'/data/RGBU/{dataset}'):
            print(f'RGBU dir not found at {dataset}')
            return
        tiffs = [f for f in os.listdir(base_path+f'/data/RGBU/{dataset}') if ((f.split('.')[-1]=='tif'))]
        if not tiffs:
            print(f'No RGBU tif files found in {dataset}')
            return
        else:
            dataset_pb = experiment_pb2.Experiment()
    
        with open(base_path+f'/data/Datasets/{dataset}/dataset.bin', mode='rb') as file:
            dataset_pb.ParseFromString(file.read())
    
        offsets = [offset for offset in dataset_pb.matched_aligned_context_images if ((offset.image.split('.')[-1] =='tif') )]
        if not offsets:
            print(f'Dataset {dataset} / {dataset_pb.title} does not contain alignment info for RGBU tiff')
            return 
        tiff = offsets[0].image
        for off in offsets:
            if ('VIS' in off.image ):
                tiff = off.image
        rgbu_array = io.imread(base_path+f'/data/RGBU/{dataset}/{tiff}')
    
    return rgbu_array

def get_datasets_with_rgbu():
    if not os.path.isdir(base_path+'/data'):
            print('Data directory not found! Run with `-r` to reload data directory')
            return                  
    dir_list = os.listdir(base_path+'/data/RGBU')
    return [ds for ds in dir_list if check_rgbu(dir_list)]

def get_pmc_rgbu_map(dataset):
    if not os.path.isfile(base_path+f'/data/Datasets/{dataset}/dataset.bin'):
        print(f'Dataset {dataset} protobuf binary not found')
        return
    
    dataset_pb = experiment_pb2.Experiment()
    
    with open(base_path+f'/data/Datasets/{dataset}/dataset.bin', mode='rb') as file:
        dataset_pb.ParseFromString(file.read())
   
    offsets = [offset for offset in dataset_pb.matched_aligned_context_images if ((offset.image.split('.')[-1] =='tif') )]
    if not offsets:
         print(f'Dataset {dataset} / {dataset_pb.title} does not contain alignment info for RGBU tiff')
         return 
    offset = offsets[0]
    for off in offsets:
        if ('VIS' in off.image ):
            offset = off


    if (offset.x_scale and offset.x_scale > 1.0) or (offset.y_scale and offset.y_scale > 1.0) :
        print(f'Scale is > 1 and I don\'t know how to handle that yet!')
        return 

    pmc_rgbu_loc_map = {}
    for loc in dataset_pb.locations:
        pmc_rgbu_loc_map[int(loc.id)] = {'i' : loc.beam.image_i-offset.x_offset, 'j': loc.beam.image_j-offset.y_offset}
    return pmc_rgbu_loc_map


def load_quant_dataframe(aggregated=True, ratios=True):
    if not os.path.isdir(base_path+'/data'):
            print('Data directory not found! Run with `-r` to reload data directory')
            return   
    quantifications = []
    for root, subdirs, files in tqdm(list(os.walk(base_path+"/data/Quantifications/")),leave=False,desc="Loading Quantification Files"):
      for f in files:
        quant = quantification_pb2.Quantification()
        if f.split('.')[-1]=='bin':
            quant_id = f.split('/')[-1].split('.')[0]
            with open(root+'/'+f, mode='rb') as file:
                quant.ParseFromString(file.read())
            with open(root+f'/summary-{quant_id}.json', mode='rb') as file:
                summary_dict = json.load(file)  
            
            for locSet in quant.locationSet:
                detector = locSet.detector
                for loc in locSet.location:
                    quant_dict={}
                    for i, label in enumerate(quant.labels):
                        if quant.types[i] == 0:
                            quant_dict[label]=loc.values[i].fvalue
                        elif quant.types[i] == 1:
                            quant_dict[label]=loc.values[i].ivalue    
                    quantifications.append({
                        'dataset':root.split('/')[-1],
                        'quant_id':quant_id,
                        'quant_name':summary_dict['params']['name'],
                        'pmc':loc.pmc,
                        'detector':locSet.detector,
                        'quant':quant_dict,
                    })
    quant_df = pd.DataFrame(quantifications)
    if aggregated:
      print(' ... Aggregating Quants ...',end='\r')
      quant_df = quant_df.groupby(['dataset','pmc']).agg({'quant':join_quants}).reset_index()
      quant_df['quant_name'] = 'Agg'
      quant_df['quant_id'] = quant_df.dataset
      print(end='\r')
    
    if ratios:
      quant_df['quant'] = quant_df.quant.apply(insert_quant_ratios)

    return quant_df

def insert_quant_ratios(quant):
    ratio_quant = quant.copy()
    for k1 in quant:
      for k2 in quant:
        if k1 != k2 and quant[k2] > 0.0:
          ratio_quant[f'{k1} / {k2}'] = quant[k1]/quant[k2]
    return ratio_quant

def join_quants(qs):
        agg_quant = {}
        for q in qs:
            for k in q:
                element = k.split('_')[0]
                if f'{element}_%' == k and f'{element}_err' in q:
                    tup = (q[f'{element}_%'],q[f'{element}_err'])
                    if tup[0] >=0 and tup[1] > 0:
                        if element in agg_quant:
                            agg_quant[element].append(tup)
                        else:
                            agg_quant[element] = [tup]
        for k in agg_quant:
            agg_quant[k] = np.average([t[0] for t in agg_quant[k]],weights=[1.0/t[1] for t in agg_quant[k]])
        return agg_quant

def quant_diff(q1,q2, custom_elements=None,diff_func=None):
    if not diff_func:
      def diff_func(a,b):
        return np.abs((a-b+0.0)/(a+b+0.0))
    
    
    q1_keys = {k for k in q1} 
    q2_keys = {k for k in q2} 
    comparison_keys = q1_keys.intersection(q2_keys)
    if custom_elements:
      if custom_elements.issubset(comparison_keys):
        return np.mean([diff_func(q1[k],q2[k]) for k in custom_elements])
      else:
        return float('NaN')
    else:
      if comparison_keys:
        return np.mean([diff_func(q1[k],q2[k]) for k in comparison_keys])
      else:
        return float('NaN')


def quant_mean(qs):
    if not qs:
        return {}
    keys = {k for k in qs[0]} 
    for q in qs[1:]:
        keys = keys.intersection({k for k in q})
    
    return {k:np.mean([q[k] for q in qs]) for k in keys}

    
        

def run_query(query_quant_dict,roi_df,quant_df,dataset_df,dropna=True,custom_elements=None, filter_funcs=None):
    results = []
    for _,r in tqdm(roi_df.iterrows(),total=len(roi_df),leave=False):
        for quant_id in quant_df[quant_df.dataset == r.dataset].quant_id.unique():
            result_quants = list(quant_df[(quant_df.dataset == r.dataset) & (quant_df.quant_id == quant_id) & (quant_df.pmc.isin(r.locationIndexes))].quant)
            
            include_quant = True
            if filter_funcs:
              for f in filter_funcs:
                try:
                  if not f(quant_mean(result_quants)):
                    include_quant = False
                except KeyError:
                  include_quant = False
            
            if include_quant:
              results.append({
                  'dataset':dataset_df[dataset_df.dataset == r.dataset].title.iloc[0],
                  'dataset_id':r.dataset,
                  'roi_key':r.roi_key,
                  'roi_name':r['name'],
                  'quant_id':quant_id,
                  'quant_name':quant_df[quant_df.quant_id == quant_id].quant_name.iloc[0],
                  'similarity':1.0-quant_diff(query_quant_dict,quant_mean(result_quants),custom_elements=custom_elements)
              })
    result_df = pd.DataFrame(results)
    if dropna:
        result_df = result_df.dropna()
    
    return result_df

