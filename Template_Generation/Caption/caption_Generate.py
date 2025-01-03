import json
import pandas as pd
from rdkit import Chem

template_list = ['SMILES',"Description"]
data_file = ""
output_file = "caption_test.json"
QA_pair = {}
qa_list = []
dataframe = pd.read_csv(data_file,nrows=100)
for i,row in dataframe.iterrows():
  temp = ""
  for name in template_list:
    if name == "SMILES":
        smiles  = Chem.MolToSmiles(Chem.MolFromSmiles(row["SMILES"]), isomericSmiles=False, canonical=True)
    if name  == "Description":
        temp = temp + f" {row[name]}"
  Question = f"How to describe this molecule {smiles}?"
  Answer = temp.strip()
  qa_pair = {"instruction": Question, "input": "", "output": Answer, "history": []}
  qa_list.append(qa_pair)
with open(output_file, 'w') as f:
    json.dump(qa_list, f, indent=4)

print("All the data has been successfully converted to JSON array format")