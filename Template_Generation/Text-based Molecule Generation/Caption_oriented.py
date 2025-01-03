import json

import pandas as pd
from rdkit import Chem

attributes = [
    "SMILES",
    "Description",
    "BalabanJ",
    "BertzCT",
    "Chi0",
    "Chi0n",
    "Chi0v",
    "Chi1",
    "Chi1n",
    "Chi1v",
    "Chi2n",
    "Chi2v",
    "Chi3n",
    "Chi3v",
    "Chi4n",
    "Chi4v",
    "ExactMolWt",
    "FpDensityMorgan1",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
    "FractionCSP3",
    "HallKierAlpha",
    "HeavyAtomCount",
    "HeavyAtomMolWt",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",
    "MaxAbsEStateIndex",
    "MaxEStateIndex",
    "MinAbsEStateIndex",
    "MinEStateIndex",
    "MolLogP",
    "MolMR",
    "MolWt",
    "NHOHCount",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRadicalElectrons",
    "NumRotatableBonds",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "NumValenceElectrons",
    "RingCount",
    "TPSA",
    "QED"
]
template_list = ['SMILES',"Description"]
data_file = "../../cpk181_all_data/tes1t.csv"
output_file = "Molecule_Generate_D_test.json"
QA_pair = {}
qa_list = []
dataframe = pd.read_csv(data_file)
for i,row in dataframe.iterrows():
  temp = ""
  for name in template_list:
    if name == "SMILES":
        smiles = row[name]
        Answer = smiles
    if name == "Description":
        temp = f"{row[name]}"
  Question = f"Can you give a Molecule SMILES and "+temp+"?"
  qa_pair = {"instruction": Question, "input": "", "output": Answer, "history": []}
  qa_list.append(qa_pair)
with open(output_file, 'w') as f:
    json.dump(qa_list, f, indent=4)

print("All the data has been successfully converted to JSON array format")