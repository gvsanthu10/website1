import numpy as np

labels = np.array(['Pancreatic ductal adenocarcinoma',
       'Neuroendocrine Tumor (insulinoma)',
       'Neuroendocrine Tumor (gastrinoma)',
       'Neuroendocrine Tumor in VHL or Zollinger Elison syndrome, MEN',
       'Neuroendocrine Tumor (non-functional)',
       'Solid Pseudopapillary Tumor (solid cystic tumor)\npapillary epithelial tumor, papillary cystic tumors)',
       'Pancreatoblastoma', 'diffuse form of pancreatic lymphoma',
       'Solitary metastases (RCC or other primaries)',
       'Metastases to the Pancreas', 'Acinar cell carcinoma',
       'Groove pancreatitis', 'Intrapancreatic Accessory Spleen',
       'Sarcoidosis', 'Castleman Disease of the Pancreas', 'lipoma'],
      dtype=object)

all_features = np.array(['small', 'Large', 'Solid', 'solid_with_necrosis', 'Single', 'Multiple', 
 'Diffuse', 'Calcification', 'fat', 'haemorrhage', 
 'duct_obst', 'Hypovascular', 'Hypervascular', 'Head', 'body', 'duodenum'])

positive = np.array([[0.91 , 0.96 , 0.98 , 0.98 , 0.11 , 0.26 , 0.06 , 1.01 , 0.56 ,
        0.66 , 0.21 , 1.01 , 1.01 , 1.01 , 0.71 , 1.01 ],
       [0.76 , 0.03 , 0.03 , 0.03 , 0.935, 0.91 , 0.96 , 0.11 , 0.66 ,
        0.56 , 0.96 , 0.01 , 0.01 , 0.01 , 0.51 , 0.01 ],
       [0.96 , 1.01 , 0.81 , 0.26 , 1.01 , 1.01 , 1.01 , 0.01 , 1.01 ,
        0.01 , 1.01 , 1.01 , 1.01 , 0.085, 0.21 , 0.985],
       [0.01 , 0.01 , 0.31 , 0.81 , 0.01 , 0.01 , 0.01 , 0.085, 0.01 ,
        0.41 , 0.01 , 0.01 , 0.01 , 0.81 , 0.46 , 0.035],
       [0.01 , 0.01 , 0.01 , 0.06 , 0.01 , 0.01 , 0.01 , 0.91 , 0.01 ,
        0.36 , 0.01 , 0.01 , 0.01 , 0.26 , 0.76 , 0.01 ],
       [0.02 , 0.36 , 0.11 , 0.11 , 0.085, 0.21 , 0.36 , 0.01 , 0.085,
        0.085, 0.31 , 0.01 , 0.01 , 0.01 , 1.01 , 0.01 ],
       [0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 ,
        0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 1.01 ],
       [0.21 , 0.01 , 0.01 , 0.01 , 0.16 , 0.21 , 0.36 , 0.01 , 0.11 ,
        0.11 , 0.36 , 0.01 , 0.01 , 0.01 , 0.085, 0.01 ],
       [0.31 , 0.01 , 0.01 , 0.01 , 0.11 , 0.01 , 0.11 , 0.16 , 0.06 ,
        0.16 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 ],
       [0.91 , 0.06 , 0.06 , 0.06 , 0.11 , 0.11 , 0.01 , 1.01 , 0.36 ,
        0.66 , 0.66 , 1.01 , 0.86 , 1.01 , 0.61 , 1.01 ],
       [0.11 , 0.96 , 0.96 , 0.96 , 0.935, 0.91 , 0.01 , 0.01 , 0.66 ,
        0.41 , 0.46 , 0.01 , 0.16 , 0.01 , 0.46 , 0.01 ],
       [0.71 , 0.96 , 0.71 , 0.76 , 0.31 , 0.01 , 0.26 , 0.01 , 0.91 ,
        0.76 , 0.36 , 1.01 , 1.01 , 0.86 , 0.66 , 0.86 ],
       [0.36 , 0.06 , 0.51 , 0.46 , 0.86 , 1.01 , 0.81 , 1.01 , 0.26 ,
        0.36 , 0.76 , 0.01 , 0.01 , 0.21 , 0.56 , 0.21 ],
       [0.71 , 0.61 , 0.61 , 0.61 , 0.86 , 0.31 , 0.76 , 0.81 , 0.26 ,
        0.56 , 0.66 , 0.01 , 0.01 , 0.56 , 0.81 , 0.61 ],
       [0.46 , 0.61 , 0.61 , 0.61 , 0.16 , 0.81 , 0.36 , 0.41 , 0.76 ,
        0.61 , 0.56 , 0.01 , 1.01 , 0.56 , 0.41 , 0.61 ],
       [0.02 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.085, 0.01 , 0.06 ,
        0.11 , 0.01 , 1.01 , 0.01 , 0.01 , 0.01 , 0.01 ]])

negative = np.array([[0.94883101, 0.94601953, 0.94489493, 0.94489493, 0.99381474,
        0.98538029, 0.99662622, 0.94320804, 0.96851139, 0.96288843,
        0.98819177, 0.94320804, 0.94320804, 0.94320804, 0.96007694,
        0.94320804],
       [0.84236884, 0.99377772, 0.99377772, 0.99377772, 0.80607219,
        0.81125743, 0.80088696, 0.97718496, 0.86310978, 0.88385072,
        0.80088696, 0.99792591, 0.99792591, 0.99792591, 0.8942212 ,
        0.99792591],
       [0.91704377, 0.91272313, 0.93000568, 0.97753269, 0.91272313,
        0.91272313, 0.91272313, 0.99913587, 0.91272313, 0.99913587,
        0.91272313, 0.91272313, 0.91272313, 0.99265492, 0.98185332,
        0.91488345],
       [0.99657287, 0.99657287, 0.89375892, 0.72240233, 0.99657287,
        0.99657287, 0.99657287, 0.97086938, 0.99657287, 0.8594876 ,
        0.99657287, 0.99657287, 0.99657287, 0.72240233, 0.84235194,
        0.98800504],
       [0.99572124, 0.99572124, 0.99572124, 0.97432745, 0.99572124,
        0.99572124, 0.99572124, 0.61063301, 0.99572124, 0.84596471,
        0.99572124, 0.99572124, 0.99572124, 0.88875229, 0.67481438,
        0.99572124],
       [0.99474142, 0.90534549, 0.97107779, 0.97107779, 0.97765102,
        0.94478487, 0.90534549, 0.99737071, 0.97765102, 0.97765102,
        0.91849195, 0.99737071, 0.99737071, 0.99737071, 0.73444152,
        0.99737071],
       [0.99265188, 0.99265188, 0.99265188, 0.99265188, 0.99265188,
        0.99265188, 0.99265188, 0.99265188, 0.99265188, 0.99265188,
        0.99265188, 0.99265188, 0.99265188, 0.99265188, 0.99265188,
        0.25784017],
       [0.95303543, 0.99776359, 0.99776359, 0.99776359, 0.96421747,
        0.95303543, 0.91948931, 0.99776359, 0.97539951, 0.97539951,
        0.91948931, 0.99776359, 0.99776359, 0.99776359, 0.98099053,
        0.99776359],
       [0.91964755, 0.99740799, 0.99740799, 0.99740799, 0.97148784,
        0.99740799, 0.97148784, 0.95852777, 0.98444791, 0.95852777,
        0.99740799, 0.99740799, 0.99740799, 0.99740799, 0.99740799,
        0.99740799],
       [0.88655104, 0.99251985, 0.99251985, 0.99251985, 0.98628639,
        0.98628639, 0.99875331, 0.87408412, 0.95511909, 0.91771834,
        0.91771834, 0.87408412, 0.8927845 , 0.87408412, 0.9239518 ,
        0.87408412],
       [0.98086021, 0.83296185, 0.83296185, 0.83296185, 0.8373118 ,
        0.84166175, 0.99826002, 0.99826002, 0.88516127, 0.92866079,
        0.91996089, 0.99826002, 0.97216031, 0.99826002, 0.91996089,
        0.99826002],
       [0.95214471, 0.93529426, 0.95214471, 0.94877462, 0.97910544,
        0.99932598, 0.98247553, 0.99932598, 0.93866435, 0.94877462,
        0.97573535, 0.93192417, 0.93192417, 0.94203444, 0.9555148 ,
        0.94203444],
       [0.96212947, 0.99368824, 0.94635008, 0.95160988, 0.90953151,
        0.89375212, 0.9147913 , 0.89375212, 0.97264906, 0.96212947,
        0.9200511 , 0.99894804, 0.99894804, 0.97790886, 0.94109028,
        0.97790886],
       [0.95851925, 0.96436161, 0.96436161, 0.96436161, 0.94975571,
        0.98188868, 0.95559807, 0.95267689, 0.98480986, 0.96728279,
        0.96144043, 0.99941576, 0.99941576, 0.96728279, 0.95267689,
        0.96436161],
       [0.97998664, 0.97346055, 0.97346055, 0.97346055, 0.99303883,
        0.96475909, 0.98433737, 0.98216201, 0.96693445, 0.97346055,
        0.97563591, 0.99956493, 0.95605762, 0.97563591, 0.98216201,
        0.97346055],
       [0.98858529, 0.99429265, 0.99429265, 0.99429265, 0.99429265,
        0.99429265, 0.95148749, 0.99429265, 0.96575588, 0.9372191 ,
        0.99429265, 0.42355723, 0.99429265, 0.99429265, 0.99429265,
        0.99429265]])

def pancreas_calculator(user_input, positive, negative, all_features, labels):
  postive_list = [1 if item in user_input else 0 for item in all_features]
  postive_array = np.array(postive_list).reshape(16,1)  #change the number g=here
  neg_array = 1- postive_array
  
  pos_multi = np.multiply(positive,postive_array)
  neg_multi = np.multiply(negative,neg_array)
  
  total_sum = pos_multi + neg_multi
  
  row_wise_sum = np.prod(total_sum, axis=0)
  
  #pre_normalize = np.multiply(row_wise_sum, prevalence)
  pre_normalize = row_wise_sum
  normalized = pre_normalize/pre_normalize.sum()
  
  list1, list2 = (list(t) for t in zip(*sorted(zip(normalized, labels))))
  
  result = {}
  for i in range(5):
    result[str(list2[::-1][i])] = round(list1[::-1][i],5)
  
  return result