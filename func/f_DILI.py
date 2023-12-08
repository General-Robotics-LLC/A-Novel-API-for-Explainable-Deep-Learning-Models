#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy
import pandas
import pandas as pd
import numpy as np
from tqdm import tqdm
# from mendeleev import get_table
from sklearn import preprocessing
from torch_geometric.data import Data
import rdkit
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import MolSurf
from rdkit.Chem.Descriptors import ExactMolWt
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import torch.nn.functional as F
import torch_geometric.nn as geonn
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
import scaffoldgraph as sg
import torch
from sklearn.metrics import r2_score
from sklearn import metrics
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol


tb_atom_feats = numpy.array(pandas.read_excel('/home/geonyeongchoi/data/mendeleev_9_features.xlsx'))
elem_feats=tb_atom_feats
    
x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}

def from_smiles2(smiles: str, with_hydrogen: bool = False,
                kekulize: bool = False,ptr = 0):
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (string, optional): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem, RDLogger

    from torch_geometric.data import Data

    RDLogger.DisableLog('rdApp.*')

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        mol = Chem.Kekulize(mol)

    xs = []
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        x.append(x_map['chirality'].index(str(atom.GetChiralTag())))        
        x.extend(elem_feats[atom.GetAtomicNum() - 1, 1:])
#         x.extend([atom.GetTotalNumHs(),atom.IsInRing()*1,atom.GetTotalDegree(),
#                          atom.GetTotalValence(),atom.GetIsAromatic()*1,
#                          (atom.GetChiralTag()==rdkit.Chem.rdchem.ChiralType.CHI_OTHER)*1,
#                         (atom.GetChiralTag()==rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)*1,
#                          (atom.GetChiralTag()==rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)*1,
#                         (atom.GetChiralTag()==rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED)*1])
            
#         x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
#         x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        x.append(x_map['degree'].index(atom.GetTotalDegree()))
        x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        x.append(x_map['num_radical_electrons'].index(
            atom.GetNumRadicalElectrons()))
        x.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(x)
        

    x = torch.tensor(np.array(xs), dtype=torch.float).view(-1, len(x))

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    extra_feature = extract_IntrinsicF(mol)

    radius=2
    nBits=1024
    ECFP6 = AllChem.GetMorganFingerprintAsBitVect(mol,radius=radius, nBits=nBits)
    ECFP = torch.tensor(numpy.float32(ECFP6)).view(1,-1)
        
    return Data(x=x, edge_index=edge_index,ECFP=ECFP, edge_attr=edge_attr,eFeature = extra_feature,smiles= smiles,ptr=ptr)
#     return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)   

# elem_feat_names = ['atomic_number','atomic_weight', 'atomic_radius', 'atomic_volume', 'dipole_polarizability',
#                    'vdw_radius', 'en_pauling','boiling_point', 'electron_affinity', 
#                    'en_allen', 'en_ghosh','mulliken_en','NumberofNeutrons']
# n_atom_feats = len(elem_feat_names)+5
n_atom_feats = 18

 
def get_elem_feats():
    #tb_atom_feats = get_table('elements')  # 118개 atom의 정보
    # elem_feats = numpy.nan_to_num(numpy.array(tb_atom_feats[elem_feat_names]))
    #return preprocessing.scale(elem_feats)
    tb_atom_feats = numpy.array(pandas.read_excel('/home/geonyeongchoi/data/mendeleev_9_features.xlsx'))
    elem_feats=tb_atom_feats
    return elem_feats

def load_dataset(path_user_dataset):
    elem_feats = get_elem_feats()
    list_mols = list()
    id_target = numpy.array(pandas.read_excel(path_user_dataset))
    for i in tqdm(range(0, id_target.shape[0])):
        mol = smiles_to_mol_graph(elem_feats, id_target[i, 0], idx=i, target=id_target[i, 1])
        if mol is not None:
            list_mols.append((id_target[i, 0], mol))
    return list_mols

        
def extract_IntrinsicF(mol, all=True):
    ## 1. Highlight Feature from CCEL 
    ## 20 EA 
    MolWt = rdkit.Chem.Descriptors.MolWt(mol)
    HeavyAtomMolWt = rdkit.Chem.Descriptors.HeavyAtomMolWt(mol)
    NumValenceElectrons = rdkit.Chem.Descriptors.NumValenceElectrons(mol)
    FractionCSP3 = rdkit.Chem.Lipinski.FractionCSP3(mol)
    HeavyAtomCount = rdkit.Chem.Lipinski.HeavyAtomCount(mol)
    NHOHCount = rdkit.Chem.Lipinski.NHOHCount(mol)
    NOCount = rdkit.Chem.Lipinski.NOCount(mol)
    NumAliphaticCarbocycles = rdkit.Chem.Lipinski.NumAliphaticCarbocycles(mol)
    NumAliphaticHeterocycles = rdkit.Chem.Lipinski.NumAliphaticHeterocycles(mol)
    NumAliphaticRings = rdkit.Chem.Lipinski.NumAliphaticRings(mol)
    NumAromaticCarbocycles = rdkit.Chem.Lipinski.NumAromaticCarbocycles(mol)
    NumAromaticHeterocycles = rdkit.Chem.Lipinski.NumAromaticHeterocycles(mol)
    NumAromaticRings = rdkit.Chem.Lipinski.NumAromaticRings(mol)
    NumHAcceptors = rdkit.Chem.Lipinski.NumHAcceptors(mol)
    NumHDonors = rdkit.Chem.Lipinski.NumHDonors(mol)
    NumHeteroatoms = rdkit.Chem.Lipinski.NumHeteroatoms(mol)
    NumRotatableBonds = rdkit.Chem.Lipinski.NumRotatableBonds(mol)
    RingCount = rdkit.Chem.Lipinski.RingCount(mol)
    MolMR = rdkit.Chem.Crippen.MolMR(mol)
    CalcNumBridgeheadAtom = rdkit.Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

    ## 2. non - Highlight Feature from CCEL 
    ## 12 EA
    ExactMolWt = rdkit.Chem.Descriptors.ExactMolWt(mol)
    NumRadicalElectrons = rdkit.Chem.Descriptors.NumRadicalElectrons(mol)
    MaxPartialCharge = rdkit.Chem.Descriptors.MaxPartialCharge(mol)
    MinPartialCharge = rdkit.Chem.Descriptors.MinPartialCharge(mol)
    MaxAbsPartialCharge = rdkit.Chem.Descriptors.MaxAbsPartialCharge(mol)
    MinAbsPartialCharge = rdkit.Chem.Descriptors.MinAbsPartialCharge(mol)
    NumSaturatedCarbocycles = rdkit.Chem.Lipinski.NumSaturatedCarbocycles(mol)
    NumSaturatedHeterocycles = rdkit.Chem.Lipinski.NumSaturatedHeterocycles(mol)
    NumSaturatedRings = rdkit.Chem.Lipinski.NumSaturatedRings(mol)
    MolLogP = rdkit.Chem.Crippen.MolLogP(mol)
    CalcNumAmideBonds = rdkit.Chem.rdMolDescriptors.CalcNumAmideBonds(mol)
    CalcNumSpiroAtoms = rdkit.Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)

    recom = [RingCount,MolWt, HeavyAtomMolWt, NumValenceElectrons, FractionCSP3, HeavyAtomCount, NHOHCount, NOCount,
             NumAliphaticCarbocycles, NumAliphaticHeterocycles, NumAliphaticRings, NumAromaticCarbocycles, 
             NumAromaticHeterocycles, NumAromaticRings, NumHAcceptors, NumHDonors, NumHeteroatoms, 
             NumRotatableBonds, MolMR, CalcNumBridgeheadAtom]
    
    if all:
        notRecom = [ExactMolWt, NumRadicalElectrons, MaxPartialCharge, MinPartialCharge,
                    MaxAbsPartialCharge, MinAbsPartialCharge, 
                    NumSaturatedCarbocycles, NumSaturatedHeterocycles, NumSaturatedRings, MolLogP, CalcNumAmideBonds, 
                    CalcNumSpiroAtoms]

        
        notRecom=numpy.nan_to_num(notRecom)
        
        recom.extend(notRecom)
        recom=numpy.float32(recom)
    return torch.tensor(recom).view(1,-1)


def load_dataset_by_name(dataset_name):

    list_mols = []
    
    if(dataset_name=='Clintox'):
        chem_dataset = numpy.array(pandas.read_excel(('/home/geonyeongchoi/data/clintox_dataset.xlsx'),header=None)) 
    elif(dataset_name=='Clintox_FDA'):
        chem_dataset = numpy.array(pandas.read_csv(('/home/geonyeongchoi/data/clintox_FDA.csv'),header=None))                  
#     elif(dataset_name=='DILI'):    
#         chem_dataset = numpy.array(pandas.read_excel(('/home/geonyeongchoi/data/dilist_test_1004.xlsx'),header=None))
    elif(dataset_name=='DILI'):
        chem_dataset = numpy.array(pandas.read_excel(('/home/geonyeongchoi/data/DILI_w_replace.xlsx'),header=None))             
    elif(dataset_name=='HIV'):
        chem_dataset = numpy.array(pandas.read_excel(('/home/geonyeongchoi/data/HIV.xlsx'),header=None))            
    elif(dataset_name=='bace'):
        chem_dataset = numpy.array(pandas.read_csv(('/home/geonyeongchoi/data/bace.csv')))
    elif(dataset_name=='BBBP'):
        chem_dataset = numpy.array(pandas.read_csv(('/home/geonyeongchoi/data/BBBP.csv')))        
    elif(dataset_name=='DH'):
        chem_dataset = numpy.array(pandas.read_excel(('/home/geonyeongchoi/data/3_CCEL_data_(wo_B).xlsx'),header=None)) 
    elif(dataset_name=='MP'):
        chem_dataset = numpy.array(pandas.read_excel(('/home/geonyeongchoi/data/MP.xlsx'),header=None)) 
    elif(dataset_name=='BP'):
        chem_dataset = numpy.array(pandas.read_excel(('/home/geonyeongchoi/data/BP.xlsx'),header=None))         
    else:
        chem_dataset = numpy.array(pandas.read_csv(('/home/geonyeongchoi/data/benchmark_data/'+dataset_name+'.csv')))
        
    for i in tqdm(range(0, chem_dataset.shape[0])):        
        if(dataset_name=='HIV'):
            if(str(chem_dataset[i,0])=='nan'):
                continue        
            
        if(chem_dataset[i, 0]==None):
            continue
            
        mol = Chem.MolFromSmiles(chem_dataset[i, 0])
        if mol is not None:
            mol = from_smiles2(chem_dataset[i, 0],with_hydrogen = True,ptr = i)        
            mol['y'] = torch.tensor(chem_dataset[i, 1], dtype=torch.float).view(1, 1)
            list_mols.append(mol)
    return list_mols
        
import torch

def train(model, optimizer, data_loader, criterion,device):
    model.train()
    train_loss = 0
    
    for i, (batch) in enumerate(data_loader):
        
        batch=batch.to(device)
        pred = model(batch)
        loss = criterion(pred,batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.detach().item()

    return train_loss / len(data_loader)

def cal_loss(model, optimizer, data_loader, criterion,device):
    model.eval()
    train_loss = 0
    
    for i, (batch) in enumerate(data_loader):
        batch = batch.to(device)
        pred = model(batch)
        loss = criterion(pred,batch.y)
        
        train_loss += loss.detach().item()

    return train_loss / len(data_loader)

def test(model, data_loader,device):
    model.eval()
    list_preds = list()

    with torch.no_grad():
        for batch in data_loader:
            batch=batch.to(device)
            preds = model(batch)
            list_preds.append(preds)

    return torch.cat(list_preds, dim=0).cpu().numpy()

def num_category(node_list,edge_list):

    node_list = np.hstack(node_list)
    edge_list = np.hstack(edge_list)

    n_list = []
    e_list = []

    df = pd.DataFrame(node_list)
    df = df[df.duplicated()==False]
    idx_num = np.array(df).reshape(-1)

    for batch_num, i in enumerate(idx_num):
        n_list.extend([batch_num]*np.sum(node_list==i))
        e_list.extend([batch_num]*np.sum(edge_list==i))
        
    return torch.tensor(n_list),torch.tensor(e_list)

def sc_batch_idx(batch):
    batch.sc_edge_idx = batch.sc_edge_idx.tolist()
    batch.sc_idx,batch.sc_edge_idx = num_category(batch.sc_idx,batch.sc_edge_idx)    
    batch_idx = 0

    begin_edge = []
    end_edge = []

    sc_begin_edge = batch.sc_begin_edge.tolist()
    sc_end_edge = batch.sc_end_edge.tolist()
    nn = []
    for j in range(len(batch.y)):
    #     n_idx = [i for i, e in enumerate(batch.sc_idx) if e == j]
    #     e_idx = [i for i, e in enumerate(batch.sc_edge_idx) if e == j]
    #     begin_edge = torch.cat((begin_edge,batch.sc_begin_edge[e_idx] + batch_idx),0)
    #     end_edge = torch.cat((end_edge,batch.sc_end_edge[e_idx] + batch_idx),0)

        n_idx = list(np.where(batch.sc_idx==j)[0])
        e_idx = list(np.where(batch.sc_edge_idx==j)[0])

        begin_edge.extend([sc_begin_edge[i]+batch_idx for i in e_idx])
        end_edge.extend([sc_end_edge[i]+batch_idx for i in e_idx])    

        nn.append(len(n_idx))
        batch_idx += len(n_idx)    

    sc_edge_index = []
    sc_edge_index.append(begin_edge)
    sc_edge_index.append(end_edge)

    batch.sc_edge_index = torch.as_tensor(sc_edge_index)    
    batch = batch.detach('sc_edge_idx')
    return batch

# def train(model, optimizer, data_loader, criterion,device):
#     model.train()
#     train_loss = 0
    
#     for i, (batch) in enumerate(data_loader):
#         batch = sc_batch_idx(batch)
#         batch=batch.to(device)
#         pred = model(batch)
#         loss = criterion(pred,batch.y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         train_loss += loss.detach().item()

#     return train_loss / len(data_loader)

# def cal_loss(model, optimizer, data_loader, criterion,device):
#     model.eval()
#     train_loss = 0
    
#     for i, (batch) in enumerate(data_loader):
#         batch = sc_batch_idx(batch)               
#         batch = batch.to(device)
#         pred = model(batch)
#         loss = criterion(pred,batch.y)
        
#         train_loss += loss.detach().item()

#     return train_loss / len(data_loader)

# def test(model, data_loader,device):
#     model.eval()
#     list_preds = list()

#     with torch.no_grad():
#         for batch in data_loader:
#             batch = sc_batch_idx(batch)
#             batch = batch.to(device)
#             preds = model(batch)
#             list_preds.append(preds)

#     return torch.cat(list_preds, dim=0).cpu().numpy()

from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

from torch_geometric.nn.inits import reset

class ADDConv(MessagePassing):
    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False, edge_dim: Optional[int] = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, 'in_features'):
                in_channels = nn.in_features
            elif hasattr(nn, 'in_channels'):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.lin = Linear(edge_dim, in_channels)

        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

    
from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

from torch_geometric.nn.inits import reset

class GNNConv(MessagePassing):
    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False, edge_dim: Optional[int] = None,aggr = 'add',
                 **kwargs):
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, 'in_features'):
                in_channels = nn.in_features
            elif hasattr(nn, 'in_channels'):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.lin = Linear(edge_dim, in_channels)

        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'
    
    
def all_stats(y_true,y_pred):
    
    acc = metrics.accuracy_score(y_true = y_true,y_pred = y_pred.round())
    bal_acc = metrics.balanced_accuracy_score(y_true = y_true,y_pred = y_pred.round())
    auroc = metrics.roc_auc_score(y_true = y_true,y_score = y_pred)
    auprc = metrics.average_precision_score(y_true = y_true,y_score = y_pred)    
#     precision = metrics.precision_score(y_true = y_true,y_pred = y_pred.round())
#     recall = metrics.recall_score(y_true = y_true,y_pred = y_pred.round()) # sensitivity
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred.round().reshape(-1)).ravel()
    sensitivity = tp/(tp+fn)
    specificity = tn/(fp+tn)
    f1 = metrics.f1_score(y_true = y_true,y_pred = y_pred.round())
    mcc = metrics.matthews_corrcoef(y_true = y_true,y_pred = y_pred.round())
    
#     stats = [auroc,auprc,acc,bal_acc,precision,recall,sensitivity,specificity,f1,mcc]  
    stats = [auroc,auprc,acc,bal_acc,f1,mcc,sensitivity,specificity]
    stats = pandas.DataFrame(stats).T
#     stats.columns = ['AUROC','AUPRC','Accuracy','Balanced_accuracy','Precision','Recall','Sensitivity','Specificity','F1','MCC']  
    stats.columns = ['AUROC','AUPRC','Accuracy','Balanced_accuracy','F1','MCC','Sens','Spec']  
    return stats
    
def mae_r2(y_true,y_pred):
    y_true = numpy.array(y_true).reshape(-1)
    
    MAE = numpy.mean(numpy.abs(y_true.reshape(-1) - y_pred.reshape(-1)))
    R2  = r2_score(y_true.reshape(-1),y_pred.reshape(-1))    
    stats = [MAE,R2]
    stats = pandas.DataFrame(stats).T
    
    stats.columns = ['MAE','R2']  
    return stats

# def atom_f(mol):
#     xs = []
#     for atom in mol.GetAtoms():
#         x = []
#         x.extend(elem_feats[atom.GetAtomicNum() - 1, :])
# #         x.append([atom.GetTotalNumHs(),
# #                   atom.IsInRing()*1,
# #                   atom.GetTotalDegree(),       
# #                   atom.GetTotalValence(),
# #                   atom.GetIsAromatic()*1,
# #                          (atom.GetChiralTag()==rdkit.Chem.rdchem.ChiralType.CHI_OTHER)*1,
# #                         (atom.GetChiralTag()==rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)*1,
# #                          (atom.GetChiralTag()==rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)*1,
# #                         (atom.GetChiralTag()==rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED)*1])
            
#         x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
#         x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
#         x.append(x_map['degree'].index(atom.GetTotalDegree()))
#         x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
#         x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
#         x.append(x_map['num_radical_electrons'].index(
#             atom.GetNumRadicalElectrons()))
#         x.append(x_map['hybridization'].index(str(atom.GetHybridization())))
#         x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
#         x.append(x_map['is_in_ring'].index(atom.IsInRing()))
#         xs.append(x)
        
#     return torch.tensor(xs, dtype=torch.float).view(-1, len(xs[0]))

# def bond_f(mol,x):
#     edge_indices, edge_attrs = [], []
#     for bond in mol.GetBonds():
#         i = bond.GetBeginAtomIdx()
#         j = bond.GetEndAtomIdx()

#         e = []
#         e.append(e_map['bond_type'].index(str(bond.GetBondType())))
#         e.append(e_map['stereo'].index(str(bond.GetStereo())))
#         e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

#         edge_indices += [[i, j], [j, i]]
#         edge_attrs += [e, e]

#     edge_index = torch.tensor(edge_indices)
#     edge_index = edge_index.t().to(torch.long).view(2, -1)
#     edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 3)
    
#     if edge_index.numel() > 0:  # Sort indices.
#         perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
#         edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
        
#     return edge_index, edge_attr        
        
# def mol_to_f(mol,with_hydrogen,kekulize):
    
#     if mol is None:
#         mol = Chem.MolFromSmiles('')
#     if with_hydrogen:
#         mol = Chem.AddHs(mol)
#     if kekulize:
#         mol = Chem.Kekulize(mol)

#     x = atom_f(mol)
#     edge_index, edge_attr = bond_f(mol,x)    
    
#     return x,edge_index, edge_attr

from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

from torch_geometric.nn.inits import reset

class GNNConv(MessagePassing):
    def __init__(self, nn: torch.nn.Module, eps: float = 0.,aggr_type = 'add',
                 train_eps: bool = False, edge_dim: Optional[int] = None, h_dim = 256,
                 **kwargs):
        kwargs.setdefault('aggr', aggr_type)
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, 'in_features'):
                in_channels = nn.in_features
            elif hasattr(nn, 'in_channels'):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.lin = Linear(edge_dim, in_channels)

        else:
            self.lin = None
        self.reset_parameters()
        
        self.lin_l = torch.nn.Linear(h_dim,h_dim)
        
    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = self.lin_l(out) +  self.nn(x_r)

        return out


    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

    
    