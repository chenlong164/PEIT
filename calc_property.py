from rdkit import Chem
import torch
import rdkit
from rdkit import RDLogger
from inspect import getmembers, isfunction
from rdkit.Chem import Descriptors
import time
from collections import OrderedDict


with open('./property_name.txt', 'r') as f:
    names = [n.strip() for n in f.readlines()][:53]

descriptor_dict = OrderedDict()
for n in names:
    if n == 'QED':
        descriptor_dict[n] = lambda x: Chem.QED.qed(x)
    else:
        descriptor_dict[n] = getattr(Descriptors, n)

#for i in getmembers(Descriptors, isfunction):
#    if i[0] in names:
#        descriptor_dict[i[0]] = i[1]
#        # print(i[0])    #name of properties
#descriptor_dict['QED'] = lambda x: Chem.QED.qed(x)


def calculate_property(smiles):
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    output = []
    for i, descriptor in enumerate(descriptor_dict):
        # print(descriptor)
        output.append(descriptor_dict[descriptor](mol))
    return torch.tensor(output, dtype=torch.float)


if __name__ == '__main__':
    st = time.time()
    output = calculate_property('CC(=O)N1CCC2(CC1)NC(=O)N(c1ccccc1)N2')
    print(output[0],output[14],output[30],output[51],output[52], output.size())
    print(time.time() - st)
    print(rdkit.__version__)

