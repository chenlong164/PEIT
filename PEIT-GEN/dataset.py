from torch.utils.data import Dataset
import torch
import random
import pandas as pd
from rdkit import Chem
import pickle
from rdkit import RDLogger
from calc_property import calculate_property
from augment import MolAugmenter
RDLogger.DisableLog('rdApp.*')

class SMILESDataset_pretrain(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        if data_length is not None:
            with open(data_path, 'r') as f:
                for _ in range(data_length[0]):
                    f.readline()
                lines = []
                for _ in range(data_length[1] - data_length[0]):
                    lines.append(f.readline())
        else:
            with open(data_path, 'r') as f:
                lines = f.readlines()
        self.data = [l.strip() for l in lines]
        with open('normalize.pkl', 'rb') as w:
            norm = pickle.load(w)
        self.property_mean, self.property_std = norm

        if shuffle:
            random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.data[index]), isomericSmiles=False, canonical=True)
        properties = (calculate_property(smiles) - self.property_mean) / self.property_std

        return properties, '[CLS]' + smiles


import csv
import pickle
from rdkit import Chem
import random

class SMILESDataset(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        self.data_path = data_path
        self.data = self.load_data(data_length, shuffle)

    def load_data(self, data_length, shuffle):
        df = pd.read_csv(self.data_path)
        if data_length:
            df = df[:data_length]
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        self.smiles = df['SMILES'].tolist()
        return df

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        smiles = self.smiles[index]
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False, canonical=True)
        return '[CLS]' + smiles

class SMILESDataset_pretrain1(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        self.data_path = data_path
        self.data = self.load_data(data_length, shuffle)
        with open('normalize.pkl', 'rb') as w:
            norm = pickle.load(w)
        self.property_mean, self.property_std = norm

    def load_data(self, data_length, shuffle):
        df = pd.read_csv(self.data_path)
        if data_length:
            df = df[:data_length]
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        self.smiles = df['SMILES'].tolist()
        self.descriptions = df['description'].tolist()
        return df

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        smiles = self.smiles[index]
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False, canonical=True)
        return '[CLS]' + smiles, '[CLS]' + self.descriptions[index]
class SMILESProCSV(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        self.data_path = data_path
        self.data = self.load_data(data_length, shuffle)
        with open('normalize.pkl', 'rb') as w:
            norm = pickle.load(w)
        self.property_mean, self.property_std = norm

    def load_data(self, data_length, shuffle):
        df = pd.read_csv(self.data_path)
        if data_length:
            df = df[:data_length]
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        self.smiles = df['SMILES'].tolist()
        return df

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        smiles = self.smiles[index]
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False, canonical=True)
        properties = (calculate_property(smiles) - self.property_mean) / self.property_std
        return  properties,'[CLS]' + smiles

class SMILESDescriptionProperties1(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        self.data_path = data_path
        self.data = self.load_data(data_length, shuffle)
        with open('normalize.pkl', 'rb') as w:
            norm = pickle.load(w)
        self.property_mean, self.property_std = norm

    def load_data(self, data_length, shuffle):
        df = pd.read_csv(self.data_path)
        if data_length:
            df = df[:data_length]
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        self.smiles = df['SMILES'].tolist()
        self.descriptions = df['description'].tolist()
        return df

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        smiles = self.smiles[index]
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False, canonical=True)
        properties = (calculate_property(smiles) - self.property_mean) / self.property_std
        return '[CLS]'+smiles, '[CLS]'+self.descriptions[index],properties
class SMILESDescriptionProperties(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        self.data_path = data_path
        self.data = self.load_data(data_length, shuffle)
        with open('normalize.pkl', 'rb') as w:
            norm = pickle.load(w)
        self.property_mean, self.property_std = norm

    def load_data(self, data_length, shuffle):
        df = pd.read_csv(self.data_path)
        if data_length:
            df = df[:data_length]
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        self.smiles = df['SMILES'].tolist()
        self.descriptions = df['description'].tolist()
        return df

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        smiles = self.smiles[index]
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False, canonical=True)
        properties = (calculate_property(smiles) - self.property_mean) / self.property_std
        return '[CLS]'+smiles, '[CLS]'+self.descriptions[index],properties

class SMILESDescription(Dataset):
    def __init__(self, data_path, data_length=None, shuffle=False):
        self.data_path = data_path
        self.data = self.load_data(data_length, shuffle)

    def load_data(self, data_length, shuffle):
        df = pd.read_csv(self.data_path)
        if data_length:
            df = df[:data_length]
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        self.smiles = df['SMILES'].tolist()
        self.descriptions = df['description'].tolist()
        return df

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        return self.smiles[index], self.descriptions[index]

