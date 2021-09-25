import numpy as np
import requests
import io

labels = np.array(['Propionic Acidemia', 'Propionic Acidemia (chronic)',
       'Methylmalonic Acidaemia (MAA)',
       'Methylmalonic Acidaemia (MAA) (chronic stge)',
       'Ethylmalonic acidemia', '3-methylglutaconic aciduria',
       'HMG-coenzyme A lyase deficiency', 'Glutaric aciduria type 1',
       'L-2-hydroxyglutaric aciduria',
       'L-2-hydroxyglutaric aciduria (chronic)',
       'D-2-hydroxyglutaric aciduria', 'Isovaleric acidemia',
       'Holocarboxylase Synthetase Deficiency',
       'β-ketothiolase deficiency', 'α-ketoglutaric aciduria',
       'Fumaric aciduria ', 'urea cycle defects',
       'Maple Syrup Urine Disease', 'Phenylketonuria',
       'Phenylketonuria (severe form)', 'Homocystinuria',
       'MTHFR (Methylenetetrahydrofolate Reductase Deficiency)',
       'Nonketotic Hyperglycinemia',
       '3-Phosphoglycerate Dehydrogenase Deficiency', 'Galactosemia',
       'Menkes Disease', 'Wilson disease',
       'Pyruvate Dehydrogenase Complex Deficiency',
       'Pyruvate Carboxylase Deficiency', 'Leigh Disease',
       'Kearns-Sayre Disease',
       'MELAS',
       'Myoclonus Epilepsy and Ragged-Red Fibers (MERRF)',
       'Leber Hereditary Optic Neuropathy (LHON)', 'Oxidation Defects',
       'glutaric aciduria type-II', 'Mucopolysaccharidoses',
       'Metachromatic Leukodystrophy', 'Multiple Sulfatase Deficiency',
       'Krabbe Disease (Globoid Cell Leukodystrophy)',
       'Krabbe Disease (Globoid Cell Leukodystrophy) (late onset)',
       'Gangliosidosis (GM2) (synonyms: Tay-Sachs (TS) disease)',
       'Niemann-Pick Disease', 'Gaucher Disease', 'Fucosidosis',
       'Mucolipidoses', 'Salla Disease', 'Zellweger Syndrome',
       'Neonatal Adrenoleukodystrophy (different from X-linked ALD)',
       'Infantile Refsum Disease', 'Rhizomelic Chondrodysplasia Punctata',
       'X-Linked Adrenoleukodystrophy',
       'X-Linked Adrenoleukodystrophy (terminal)',
       'Adrenomyeloneuropathy', 'Canavan Disease',
       'Megalencephalic Leukoencephalopathy with Subcortical Cysts (Van Der Knaap Disease)',
       'Vanishing White Matter Disease', 'Alexander Disease',
       'Leukodystrophy with Brainstem and Spinal Cord Involvement and High Lactate',
       'Aicardi-Goutières Syndrome', 'Cockayne Disease',
       'Pelizaeus-Merzbacher Disease', 'Carbonic Anhydrase II Deficiency',
       'Persistent Hyperinsulinemic Hypoglycemia (Nesidioblastosis)',
       'Creatine Deficiency',
       'Leukoencephalopathy Associated with Polyol Metabolism Abnormality',
       'Biotin-Responsive Encephalopathies',
       'Cerebrotendinous Xanthomatosis', 'Sjögren-Larsson Syndrome',
       'Hypoxic-ischemic encephalopathy (mild hypotensive type)',
       'Hypoxic-ischemic encephalopathy (profound type)',
       'Multicystic Encephalomalacia',
       'Ischemic Infarction in the Newborn',
       'Periventricular Leukomalacia (PVL)',
       'Periventricular Leukomalacia (PVL)',
       'Germinal Matrix Hemorrhage-Intraventricular Hemorrhage (GMH-IVH)',
       'Germinal Matrix Hemorrhage-Intraventricular Hemorrhage (GMH-IVH) (complicated)',
       'Neuronal Ceroid Lipofuscinosis',
       'Alpers Disease (Alpers-Huttenlocher syndrome)',
       'Pantothenate Kinase-Associated Neurodegeneration (Hallervorden-Spatz Syndrome)',
       'Juvenile-onset Huntington’s Disease', 'Cerebellar Atrophy',
       'Infantile Neuroaxonal Dystrophy',
       'Autosomal Dominant Cerebellar Ataxia (ADCA)',
       'Dentatorubral-Pallidoluysian Atrophy (DRPLA)',
       'Friedreich’s Ataxia', 'Hypertrophic olivary degeneration',
       'Multiple sclerosis', 'Tumefactive demyelinating disease',
       'Schilder’s Disease', 'Balò’s Concentric Sclerosis',
       'ADEM (Acute Disseminated Encephalomyelitis)',
       'Acute Hemorrhagic Encephalomyelitis and Acute Necrotic Encephalomyelitis',
       'Cytomegalovirus Infection', 'Toxoplasmosis',
       'Congenital Rubella Infection',
       'Congenital Human Immunodeficiency Virus (HIV) infection',
       'Congenital Human Immunodeficiency Virus (HIV) infection (severe)',
       'Progressive multifocal leukoencephalopathy (PML)',
       'Neonatal Herpes Simplex Virus Infection',
       'Neonatal Herpes Simplex Virus Infection (severe)',
       'Congenital Syphilis', 'Congenital Varicella',
       'Neonatal Bacterial Leptomeningitis ',
       'Neonatal Bacterial Leptomeningitis with ventriculitis',
       'Neonatal Bacterial Leptomeningitis with vasculitis', 'Cerebritis',
       'Tuberculous meningitis', 'Tuberculous meningitis with vasculitis',
       'Lyme Disease (Neuroborreliosis)',
       'Lyme Disease (Neuroborreliosis) complicated with vasculitis',
       'Herpes Simplex Virus Encephalitis',
       'Human Herpesvirus-6 Infection', 'Chickenpox',
       'Acute Cerebellitis', 'Influenza Virus Infection',
       'Epstein-Barr Virus (EBV) Infection',
       'Subacute Sclerosing Panencephalitis',
       'Subacute Sclerosing Panencephalitis (severe)',
       'Brainstem encephalitis', 'Rasmussen’s Encephalitis',
       'Reye Syndrome', 'Cryptococcus infection', 'Toxocaral Disease',
       'Hypermethioninemic States', 'Rett syndrome', 'Fabry disease',
       'Autosomal recessive spastic ataxia of Charlevoix', 'Kernicterus',
       'Neurological cretinism',
       'Radiation injury (if there is history of radiotherapy) ',
       'Mineralizing microangiopathy in radiation injury (if history of radiotherapy) ',
       'Sickle cell disease',
       'Cystic leukoencephalopathy without megalencephaly',
       'HHV-6 encephalitis', 'Cortical tubers of TS',
       'Tuberous Sclerosis', '18q syndrome', 'Jacobsen syndrome',
       'Trichothiodystrophy with Dysmyelination ', 'Hypomelanosis of ito',
       'CACH (Childhood Ataxia with Central Nervous System Hypomyelination/Vanishing White Matter)',
       'Oculo-cerebro-renal syndrome of Lowe (OCRL)',
       'Sturge Weber Syndrome',
       'Capillary telangiectasis (think of hereditary telangiectasia if multiple lesions)',
       'Encephalitis lethargica',
       'Focal areas of signal intensity (FASI) in NF1',
       'Mild encephalitis/encephalopathy with reversible splenial lesion (MERS)',
       'Bilateral Parieto-occipital Calcifications with Celiac disease',
       'Susac syndrome', 'Migraine', 'Hemophagocytic lymphohistiocytosis',
       'Metronidazole toxicity', 'Vasculitis', 'Hypoxic ischemic injury'],
      dtype=object)

all_features = np.array(['neonate', 'infant', 'juvenile', 'solitary', 'graymatter', 'wavy',
       'oval', '2cm', 'symmetrical', 'butterfly', 'frontotemporal',
       'parietooccipital', 'csf', 'cc', 'cerebellar', 'small_cyst',
       'enhacement', 'diffusion', 'hippocampus', 'subcortical_cal',
       'small_cal', 'laminar', 'post_ic', 'pituitary', 'Hemorrhage',
       'Meningeal', 'basilar_meningeal', 'Ependymal', 'Subependymal',
       'Moyamoya', 'vrs', 'atrophy', 'vermial', 'csf_cyst', 'hydro',
       'Sylvian', 'germinolytic', 'migration', 'symmetrical_bg',
       'Discrete_bg', 'Putamen', 'putamen_t1', 'Globus', 'globus_t1',
       'Thalamus', 'thalamus_t1', 'Caudate', 'caudate_atrophy', 'Tiger',
       'diffusion_bg', 'Periaquiductal', 'pandas', 'Cerebral_peduncles',
       'small_cerebellar_penduncle', 'substantia_nigra', 'red_nuc',
       'pons', 'central_pons', 'Tegmentum', 'Dentate', 'ion',
       'medulla_oblongata', 'diffusion_bstem'])

prevalence = np.array([0.00561782, 0.00561782, 0.00674138, 0.00674138, 0.00505604,
       0.00505604, 0.00505604, 0.00505604, 0.00505604, 0.00505604,
       0.00505604, 0.00505604, 0.00505604, 0.00505604, 0.00505604,
       0.00505604, 0.00674138, 0.00674138, 0.00561782, 0.00561782,
       0.00449426, 0.00449426, 0.00674138, 0.00449426, 0.00505604,
       0.00674138, 0.00898851, 0.00674138, 0.00674138, 0.01067386,
       0.00786495, 0.00898851, 0.00561782, 0.00561782, 0.00561782,
       0.00674138, 0.00786495, 0.00898851, 0.00561782, 0.00898851,
       0.00898851, 0.00674138, 0.00561782, 0.00561782, 0.00561782,
       0.00561782, 0.00561782, 0.00786495, 0.00674138, 0.0035954 ,
       0.00382012, 0.00898851, 0.00898851, 0.00674138, 0.00842673,
       0.00898851, 0.00561782, 0.00898851, 0.00674138, 0.00337069,
       0.00561782, 0.00668521, 0.00561782, 0.00561782, 0.00449426,
       0.00393247, 0.00505604, 0.00505604, 0.00505604, 0.00898851,
       0.00898851, 0.00561782, 0.00898851, 0.00955029, 0.00955029,
       0.00898851, 0.00674138, 0.00561782, 0.00561782, 0.00561782,
       0.00561782, 0.00561782, 0.00561782, 0.00561782, 0.00561782,
       0.00561782, 0.00561782, 0.00393247, 0.00393247, 0.00393247,
       0.00449426, 0.00842673, 0.00674138, 0.00786495, 0.00786495,
       0.00786495, 0.00786495, 0.00786495, 0.00674138, 0.00786495,
       0.00786495, 0.00674138, 0.00674138, 0.00898851, 0.00674138,
       0.00674138, 0.00730317, 0.00730317, 0.00730317, 0.00561782,
       0.00561782, 0.00786495, 0.0061796 , 0.0061796 , 0.00561782,
       0.00674138, 0.00674138, 0.00786495, 0.00786495, 0.00674138,
       0.00730317, 0.00674138, 0.00674138, 0.00674138, 0.00674138,
       0.00674138, 0.00730317, 0.00674138, 0.00786495, 0.00449426,
       0.00674138, 0.00674138, 0.00730317, 0.00786495, 0.00730317,
       0.00898851, 0.00898851, 0.00561782, 0.00561782, 0.00561782,
       0.00561782, 0.00449426, 0.00449426, 0.00674138, 0.00674138,
       0.00505604, 0.00674138, 0.00561782, 0.00449426, 0.00674138,
       0.00786495, 0.00449426, 0.00505604, 0.00668521, 0.00620769])

positive_url = "https://raw.githubusercontent.com/gvsanthu10/website/main/weights/peads_positve.npy"
negative_url = 'https://raw.githubusercontent.com/gvsanthu10/website/main/weights/peads_negative.npy'

#loading positive weights
response = requests.get(positive_url)
response.raise_for_status()
positive = np.load(io.BytesIO(response.content))

#loading negative weights
response = requests.get(negative_url)
response.raise_for_status()
negative = np.load(io.BytesIO(response.content))

def peads_calculator(user_input, positive, negative, all_features, labels, prevalence):
  postive_list = [1 if item in user_input else 0 for item in all_features]
  postive_array = np.array(postive_list).reshape(63,1)  #change the number g=here
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