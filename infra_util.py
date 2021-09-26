import pandas as pd
import numpy as np

import requests
import io

labels = np.array(['brainstem glioma', 'pilocytic astrocytoma', 'Glioblastoma (GBM)',
                   'ganglioglioma',
                   'Dysplastic cerebellar gangliocytoma (Lhermitte-Duclos disease (LDD)',
                   'atypical  teratoid-rhabdoid tumor', 'teratoma', 'abscess',
                   'tubercloma ',
                   'Focal areas of signal intensity (unidentified bright objects)',
                   'anaplastic astrocytoma ', 'gliomatosis cerebri ',
                   'oligodendroglioma ', 'Hemangioblastoma', 'Lymphoma ',
                   'radiation necrosis (if there is a history of radiotherapy)',
                   'cavernoma (or cavernoma complicated with hemorrhage)',
                   'solitary Metastasis', 'cerebellar liponeurocytoma',
                   'pontine hemorrhage', 'hemorrhage (hematoma)',
                   'Rosette-Forming Glioneuronal Tumor', 'medulloblastoma (lateral)',
                   'toxoplasmosis', 'Metastasis', 'kaposi sarcoma',
                   'NCC (neurocysticercosis)'], dtype=object)


all_features = np.array(['Child', 'Adult', 'solitary', 'multiple', 'solid_with_necrosis',
                         'solid_cystic', 'cystic', 'caseating', 'serpentine', 'whole_lesion_enhace', 'Enhanced_wall', 'hemorrhage',
                         'Popcorn', 'Calcification', 'fat', 'Diffusion_solid', 'Diffusion_cavity',
                         'wm_edema', 'bg_involv', 'peripheral', 'brainstem', 'striated'], dtype=object)

prevalence = np.array([0.5904059, 0.66420664, 0.5904059, 0.29520295, 0.29520295,
                       0.25830258, 0.29520295, 0.29520295, 0.29520295, 0.29520295,
                       0.22140221, 0.22140221, 0.22140221, 0.5904059, 0.51660517,
                       0.22140221, 0.5904059, 0.5904059, 0.29520295, 0.22140221,
                       0.22140221, 0.29520295, 0.5904059, 0.29520295, 0.51660517,
                       0.22140221, 0.29520295])

positive_url = "https://raw.githubusercontent.com/gvsanthu10/website/main/weights/infra_positve.npy"
negative_url = 'https://raw.githubusercontent.com/gvsanthu10/website/main/weights/infra_negative.npy'

# loading positive weights
response = requests.get(positive_url)
response.raise_for_status()
positive = np.load(io.BytesIO(response.content))

# loading negative weights
response = requests.get(negative_url)
response.raise_for_status()
negative = np.load(io.BytesIO(response.content))


def infra_intra_axial_calculator(user_input, positive, negative, all_features, labels, prevalence):
    postive_list = [1 if item in user_input else 0 for item in all_features]
    postive_array = np.array(postive_list).reshape(
        22, 1)  # change the number g=here
    neg_array = 1 - postive_array

    pos_multi = np.multiply(positive, postive_array)
    neg_multi = np.multiply(negative, neg_array)

    total_sum = pos_multi + neg_multi

    row_wise_sum = np.prod(total_sum, axis=0)

    pre_normalize = np.multiply(row_wise_sum, prevalence)
    normalized = pre_normalize/pre_normalize.sum()

    list1, list2 = (list(t) for t in zip(*sorted(zip(normalized, labels))))

    result = {}
    for i in range(5):
        result[str(list2[::-1][i])] = round(list1[::-1][i], 5)

    return result
