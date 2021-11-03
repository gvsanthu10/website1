import numpy as np

labels = np.array(['Pancreatic pseudocyst', 'serous cystadenoma',
       'mucinous cystic neoplasm',
       'IPMN (intraductal papillary neoplasms)',
       'Pancreatic intraductal tubulopapillary neoplasm'], dtype=object)

all_features = np.array(['cystic', 'honey', 'calcification', 'haemo', 'small', 
                         'Large', 'head', 'body', 'duodenum', 'Single', 
                         'multiple', 'intraductal', 'Debris'], dtype=object)

prevalence = np.array([2.5, 2, 2, 2, 1.8])

positive = np.array([[1.01 , 0.61 , 0.86 , 0.56 , 0.71 ],
       [0.01 , 0.56 , 0.11 , 0.36 , 0.31 ],
       [0.01 , 0.41 , 0.26 , 0.01 , 0.02 ],
       [0.11 , 0.085, 0.11 , 0.085, 0.035],
       [0.56 , 0.86 , 0.21 , 0.56 , 0.56 ],
       [0.71 , 0.36 , 0.86 , 0.36 , 0.46 ],
       [0.51 , 0.86 , 0.26 , 0.81 , 0.71 ],
       [0.66 , 0.16 , 0.86 , 0.36 , 0.71 ],
       [0.035, 0.085, 0.015, 0.015, 0.11 ],
       [0.86 , 0.86 , 1.01 , 0.56 , 0.26 ],
       [0.11 , 0.085, 0.06 , 0.46 , 0.76 ],
       [0.01 , 0.01 , 0.01 , 0.985, 0.985],
       [0.41 , 0.06 , 0.085, 0.06 , 0.985]])

negative = np.array([[0.98498054, 0.99092884, 0.98721115, 0.99167238, 0.98944177],
       [0.99805232, 0.89092996, 0.97857553, 0.92988355, 0.93962194],
       [0.99562681, 0.82069927, 0.8862971 , 0.99562681, 0.99125362],
       [0.9958726 , 0.99681065, 0.9958726 , 0.99681065, 0.99868674],
       [0.97311272, 0.95870882, 0.98991727, 0.97311272, 0.97311272],
       [0.97138562, 0.9854913 , 0.96534033, 0.9854913 , 0.98146111],
       [0.97805011, 0.96298646, 0.98880986, 0.96513841, 0.96944231],
       [0.94990781, 0.98785644, 0.93472836, 0.97267699, 0.94611295],
       [0.99389149, 0.98516504, 0.99738207, 0.99738207, 0.98080182],
       [0.9561195 , 0.9561195 , 0.94846592, 0.97142665, 0.9867338 ],
       [0.97156177, 0.978025  , 0.98448824, 0.88107648, 0.80351766],
       [0.99482854, 0.99482854, 0.99482854, 0.49061146, 0.49061146],
       [0.8574212 , 0.97913481, 0.97044098, 0.97913481, 0.65746313]])

def pancreas_cystic_calculator(user_input, positive, negative, all_features, labels, prevalence):
  postive_list = [1 if item in user_input else 0 for item in all_features]
  postive_array = np.array(postive_list).reshape(13,1)  #change the number g=here
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