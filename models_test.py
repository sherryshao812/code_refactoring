from models import *
import dgl
from data import load_data
from tqdm import tqdm

GCN_Model = GCNStoModel_MultiHead(in_feats=1433, hid_feats=64, out_feats=7, num_task=3, pred_head_out=3)
SAGE_Model = SAGEStoModel_MultiHead(in_feats=1433, hid_feats=64, out_feats=7, num_task=3, pred_head_out=3)
GAT_Model = GATStoModel_MultiHead(in_feats=1433, hid_feats=64, out_feats=7, num_task=3, num_heads=2, pred_head_out=3)

print(GCN_Model.float())
print(SAGE_Model.float())
print(GAT_Model.float())
