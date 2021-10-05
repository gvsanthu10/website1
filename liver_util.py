import numpy as np
import requests
import io

labels = np.array(['Simple liver cyst', 'Caroli disease', 'regenerating nodule(s)',
       'dysplastic nodule(s)', 'lipoma', 'myolipoma',
       'bile duct hamartoma',
       'hypovascular metastasis (colon, lung, prostate, TCC,,,etc)',
       'lymphoma', 'sarcoidosis', 'histoplasmosis', 'focal fat',
       'Confluent Hepatic Fibrosis', 'Hepatic adenoma',
       'HCC (hepatocellular carcinoma)',
       'hypervascular metastasis (islet cell tumor, breast ca, carcinoid,,,etc)',
       'FNH (focal nodular hyperplasia)', 'Fibrolamellar HCC',
       'liver abscess', 'microabscesses/fungal infection',
       'Radiofrequency-ablated Areas',
       'Biliary Cystadenoma or Cystadenocarcinoma.',
       'Intraductal papillary neoplasm of the bile ducts\n',
       'haematoma (if abdominal trauma exists)', 'haemangioma',
       'cholangiocarcinoma', 'Peliosis Hepatis',
       'hepatic mesenchymal hamartoma', 'hepatoblastoma',
       'Undiffereniated embryonal sarcoma', 'hemangioendothelioma',
       'Hydatid cyst', 'THID (Transient Hepatic Intensity Difference)',
       'Focal hypereosinophilic infiltration (FEI)',
       'Post-operative changes (pseudolesion/ometum fat)',
       'adrenal rest tumor', 'pseudolipoma of Glisson capsule',
       'Xanthomatous lesion in LCH (Langerhand Cell Histocytosis) ',
       'Hepatic amyloidosis', 'haemtochromatosis/Wilson Disease'],
      dtype=object)

all_features = np.array(['Child', 'Adult', 'Solid', 'solid_with_necrosis', 'multiple', 
                  'cystic', 'thin_walled_cyst', 'thin_cyst_with_biliary', 'daughter_cysts', 
                  'lowT1', 'intermediateT1', 't2dark', 'lowT2', 't2high', 'fat', 'Calcification', 
                  'early_prog_filling', 'early_intense', 'rapid_wash', 'gradual', 'mild_enhacement', 
                  'marginal_enhace', 'central_scar_enhance', 'central_scar_no', 
                  'retained_contrast', 'Isointese_delayed', 'Diffusion', 'Wedge', 
                  'Retraction', 'cirrhosis', 'other', 'biliary_dilatation'])

prevalence = np.array([1.  , 0.3 , 0.8 , 0.8 , 0.6 , 0.6 , 0.5 , 1.  , 0.3 , 0.5 , 0.5 ,
       0.7 , 0.45, 1.  , 1.5 , 1.  , 1.  , 0.4 , 0.85, 0.6 , 0.6 , 0.6 ,
       0.5 , 0.5 , 1.  , 0.8 , 0.5 , 0.5 , 0.65, 0.5 , 0.7 , 0.5 , 0.8 ,
       0.5 , 0.2 , 0.45, 0.5 , 0.5 , 0.5 , 0.5 ])

positive_url = "https://raw.githubusercontent.com/gvsanthu10/website/main/weights/liver_positve.npy"
negative_url = "https://raw.githubusercontent.com/gvsanthu10/website/main/weights/liver_negative.npy"

#loading positive weights
response = requests.get(positive_url)
response.raise_for_status()
positive = np.load(io.BytesIO(response.content))

#loading negative weights
response = requests.get(negative_url)
response.raise_for_status()
negative = np.load(io.BytesIO(response.content))

def liver_calculator(user_input, positive, negative, all_features, labels, prevalence):
  postive_list = [1 if item in user_input else 0 for item in all_features]
  postive_array = np.array(postive_list).reshape(32,1)  #change the number g=here
  neg_array = 1- postive_array
  
  pos_multi = np.multiply(positive,postive_array)
  neg_multi = np.multiply(negative,neg_array)
  
  total_sum = pos_multi + neg_multi
  
  row_wise_sum = np.prod(total_sum, axis=0)
  
  pre_normalize = np.multiply(row_wise_sum, prevalence)
  normalized = pre_normalize/pre_normalize.sum()
  
  list1, list2 = (list(t) for t in zip(*sorted(zip(normalized, labels))))
  
  result = {}
  for i in range(5):
    result[str(list2[::-1][i])] = round(list1[::-1][i],5)
  
  return result