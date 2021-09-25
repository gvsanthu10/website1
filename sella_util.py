import numpy as np
import requests
import io

labels = np.array(['Pituitary macroadenoma', 'Lymphocytic hypophysitis',
       'Langerhans cell histocytosis', 'Neurosarcoidosis',
       'Tuber cinereum hamartoma ', 'Rathke cleft cyst',
       'Adamantinomatous craniopharyngioma',
       'Papillary craniopharyngioma', 'Pituitary apoplexy', 'Pituicytoma',
       'Spindle cell oncocytoma', 'Granular cell tumor',
       'Hypothalamic glioma', 'Clival chorodoma', 'Chondrosarcoma',
       'Extramedullary hematopiosis', 'Meningioma', 'Atypical meningioma',
       'Hemangiopericytoma', 'Meningeal metastasis', 'Dermoid cyst',
       'Epidermoid cyst', 'Germinoma', 'Hemangioma of the cavenous sinus',
       'Lipoma', 'Racemose neurocysticercosis (NCC)',
       'Meningeal melanocytoma',
       'Non-germinoma germ cell tumors (teratoma, mixed GCT)',
       'Cholesterol cyst', 'IgG4-related disease', 'Pituitary abscess',
       'Central diabetes insipidus', 'Arachnoid cyst',
       'Tolosa-Hunt syndrome '], dtype=object)

all_features = np.array(['solid', 'cystic', 'grape', 
                  'inseperable', 'thickstack', 'lostpostspot', 'tuber', 'parasellar', 'Homogeneous', 'Heterogeneous', 
                  'Haemorrhagic', 'Calcification', 'Diffusion', 'fat', 'chiasm', 'cavernous', 'Perivascular', 'pineal', 
                  'Leptomeningeal', 'Dural', 'remodeling', 'hyperostosis', 'Thumb'])

prevalence = np.array([1.5 , 1.  , 1.  , 0.75, 0.9 , 1.  , 0.9 , 0.9 , 0.9 , 0.8 , 0.8 ,
       1.  , 1.  , 1.  , 1.  , 0.8 , 1.  , 0.75, 0.9 , 1.  , 1.  , 1.  ,
       1.  , 1.  , 1.  , 1.  , 0.7 , 1.  , 1.  , 1.  , 0.7 , 1.  , 1.  ,
       1.  ])

#fetching the weights from github
positive_url = "https://raw.githubusercontent.com/gvsanthu10/website/main/weights/sella_positve.npy"
negative_url = 'https://raw.githubusercontent.com/gvsanthu10/website/main/weights/sella_negative.npy'

#loading positive weights
response = requests.get(positive_url)
response.raise_for_status()
positive = np.load(io.BytesIO(response.content))

#loading negative weights
response = requests.get(negative_url)
response.raise_for_status()
negative = np.load(io.BytesIO(response.content))

def sella_calculator(user_input, positive, negative, all_features, labels, prevalence):
  postive_list = [1 if item in user_input else 0 for item in all_features]
  postive_array = np.array(postive_list).reshape(23,1)  #change the number g=here
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