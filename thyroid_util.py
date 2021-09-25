import numpy as np

pos_weights = np.array([[0.94 , 0.89 ],
       [0.14 , 0.34 ],
       [0.425, 0.84 ],
       [0.59 , 0.24 ],
       [0.19 , 0.49 ],
       [0.79 , 0.49 ],
       [0.29 , 0.66 ],
       [0.065, 0.14 ],
       [0.59 , 0.19 ],
       [0.59 , 0.89 ],
       [0.39 , 0.09 ],
       [0.69 , 0.94 ],
       [0.29 , 0.14 ],
       [0.22 , 0.85 ],
       [0.75 , 0.4  ]])

neg_weights = np.array([[0.99949334, 0.99952029],
       [0.98192098, 0.95609382],
       [0.9663859 , 0.93356273],
       [0.92190112, 0.96823096],
       [0.97238308, 0.92877741],
       [0.96840301, 0.98040186],
       [0.96741252, 0.92583539],
       [0.99357604, 0.98616378],
       [0.88256574, 0.96218219],
       [0.98239098, 0.97343724],
       [0.88152275, 0.9726591 ],
       [0.98824494, 0.98398586],
       [0.97400084, 0.98744868],
       [0.94125751, 0.77304039],
       [0.94905573, 0.97282972]])

all_features = ['solid', 'partly_solid','wide','tall','hyper','hypo',
            'nocal','macro','micro','well','ill','peri','noperi','perivasc','intervasc']
labels = ['malignant', 'benign']

def thyroid_cal(pos_weights, neg_weights, user_input, labels):
  #Creating postive and negative input rows(columns actually)
  postive_input = [1 if feature in user_input else 0 for feature in all_features]
  postive_input = np.array(postive_input).reshape(15,1)
  negative_input = 1- postive_input

  #elememtwise multiplicaiton postive row with postivie matrix and same for negative
  pos_multi = np.multiply(pos_weights,postive_input)
  neg_multi = np.multiply(neg_weights,negative_input)

  #Combine two matixes
  total_sum = pos_multi + neg_multi

  #elementwise sum
  row_wise_prod = np.prod(total_sum, axis=0)

  #normalize
  row_wise_sum = row_wise_prod/row_wise_prod.sum()

  #Sort
  list1, list2 = (list(t) for t in zip(*sorted(zip(row_wise_sum, labels))))

  result = f"The nodule being {list2[1]} with score {round(list1[1], 2)} and being {list2[0]} with score {round(list1[0], 2)}"
  return result