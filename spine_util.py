import numpy as np

all_features = ['convex', 'posterior', 'epi', 'paraspinal', 'destruc', 'Fluid', 'normal_marrow', 'retro', 'mlutpl', 'degene', 'scnodules', 't12', 'disc', 'other']

positive = np.array([[0.85 , 0.075],
       [0.8  , 0.075],
       [0.65 , 0.175],
       [0.7  , 0.175],
       [0.75 , 0.15 ],
       [0.075, 0.75 ],
       [0.1  , 0.55 ],
       [0.05 , 0.9  ],
       [0.6  , 0.75 ],
       [0.075, 0.65 ],
       [0.1  , 0.8  ],
       [0.25 , 0.7  ],
       [0.11 , 0.65 ],
       [0.7  , 0.05 ]])

negative = np.array([[0.49508414, 0.9554486 ],
       [0.53760357, 0.95665033],
       [0.83458561, 0.95546536],
       [0.80534882, 0.95133721],
       [0.73751668, 0.94750334],
       [0.95796254, 0.57962541],
       [0.96193825, 0.79066036],
       [0.96487397, 0.36773151],
       [0.99464303, 0.99330379],
       [0.96098761, 0.66189259],
       [0.95032606, 0.60260849],
       [0.95786795, 0.88203026],
       [0.95561628, 0.73773256],
       [0.54735546, 0.96766825]])

labels = np.array(['Metastatic', 'Osteoporotic'])

def spine_calculator(user_input, positive, negative, all_features, labels):
  postive_list = [1 if item in user_input else 0 for item in all_features]
  postive_array = np.array(postive_list).reshape(14,1)  #change the number g=here
  neg_array = 1- postive_array
  
  pos_multi = np.multiply(positive,postive_array)
  neg_multi = np.multiply(negative,neg_array)
  
  total_sum = pos_multi + neg_multi
  
  pre_normalize = np.prod(total_sum, axis=0)
  
  normalized = pre_normalize/pre_normalize.sum()
  
  tuple_list = list(zip(normalized, labels))
  
  result = {}
  for i in range(2):
    result[tuple_list[i][1]] = tuple_list[i][0]
  
  return result
