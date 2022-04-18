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


# sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
# graph, data_info = load_data('cora')
# train_list = list(range(40))
# device = 'cpu'
#
# train_dataloader = dgl.dataloading.NodeDataLoader(graph, train_list, sampler,
#   device=device, batch_size=1, shuffle=True, drop_last=False, num_workers=0
# )
#
# with tqdm(train_dataloader) as tq:
#     for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
#         print(step, (input_nodes, output_nodes, mfgs))
