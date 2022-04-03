import torch

import scipy.sparse as sp
import torch.nn.functional as F
import numpy as np



def cmd(X, X_test, K=5,bound=2):
	"""
	central moment discrepancy (cmd)
	"""
	x1 = X
	x2 = X_test
	mx1 = x1.mean(0)
	mx2 = x2.mean(0)
	
	sx1 = x1 - mx1
	sx2 = x2 - mx2

	dm = l2diff(mx1,mx2)/bound
	scms = [dm]
	for i in range(K-1):
		scms.append(moment_diff(sx1,sx2,i+2)/(bound**(i+2)))
	return sum(scms)

def l2diff(x1, x2):
	"""
	standard euclidean norm
	"""
	return (x1-x2).norm(p=2)

def moment_diff(sx1, sx2, k):
	"""
	difference between moments
	"""
	ss1 = sx1.pow(k).mean(0)
	ss2 = sx2.pow(k).mean(0)
	return l2diff(ss1,ss2)




def cross_entropy(x, labels):
	#epsilon = 1 - math.log(2)
	y = F.cross_entropy(x, labels.view(-1), reduction="none")
	#y = torch.log(epsilon + y) - math.log(epsilon)
	return torch.mean(y)

def pairwise_distances(x, y=None):
	'''
	Input: x is a Nxd matrix
		   y is an optional Mxd matirx
	Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
			if y is not given then use 'y=x'.
	i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
	'''
	x_norm = (x**2).sum(1).view(-1, 1)
	if y is not None:
		y_t = torch.transpose(y, 0, 1)
		y_norm = (y**2).sum(1).view(1, -1)
	else:
		y_t = torch.transpose(x, 0, 1)
		y_norm = x_norm.view(1, -1)
	
	dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
	#dist = torch.mm(x, y_t)
	#Ensure diagonal is zero if x=y
	#if y is None:
	#     dist = dist - torch.diag(dist.diag)
	return torch.clamp(dist, 0.0, np.inf)

def naiveIW(X, Xtest, _A=None, _sigma=1e1):
	prob =  torch.exp(- _sigma * torch.norm(X - Xtest.mean(dim=0), dim=1, p=2) ** 2 )
	for i in range(_A.shape[0]):
		prob[_A[i,:]==1] = F.normalize(prob[_A[0,:]==1], dim=0, p=1) * _A[i,:].sum()
	return prob

def MMD(X,Xtest):
	H = torch.exp(- 1e0 * pairwise_distances(X)) + torch.exp(- 1e-1 * pairwise_distances(X)) + torch.exp(- 1e-3 * pairwise_distances(X))
	f = torch.exp(- 1e0 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(X, Xtest))
	z = torch.exp(- 1e0 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(Xtest, Xtest))
	MMD_dist = H.mean() - 2 * f.mean() + z.mean()
	return MMD_dist

def KMM(X,Xtest,_A=None, _sigma=1e1):
	#embed()
	if False:
		H = X.matmul(X.T)
		f = X.matmul(Xtest.T)
		z = Xtest.matmul(Xtest.T)

	else:
		H = torch.exp(- 1e0 * pairwise_distances(X)) + torch.exp(- 1e-1 * pairwise_distances(X)) + torch.exp(- 1e-3 * pairwise_distances(X))
		f = torch.exp(- 1e0 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(X, Xtest))
		z = torch.exp(- 1e0 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(Xtest, Xtest))
		H /= 3
		f /= 3

	MMD_dist = H.mean() - 2 * f.mean() + z.mean()
	
	nsamples = X.shape[0]
	f = - X.shape[0] / Xtest.shape[0] * f.matmul(torch.ones((Xtest.shape[0],1)))
	#eps = (math.sqrt(nsamples)-1)/math.sqrt(nsamples)
	eps = 10

	G = - np.eye(nsamples)
	#h = np.zeros((nsamples,1))
	#if _A is None:
	#    return None, MMD_dist
	#A = 
	b = np.ones([_A.shape[0],1]) * 20
	h = - 0.2 * np.ones((nsamples,1))
	
	from cvxopt import matrix, solvers
	#return quadprog.solve_qp(H.numpy(), f.numpy(), qp_C, qp_b, meq)
	try:
		solvers.options['show_progress'] = False
		sol=solvers.qp(matrix(H.numpy().astype(np.double)), matrix(f.numpy().astype(np.double)), matrix(G), matrix(h), matrix(_A), matrix(b))
	except:
		embed()
	#embed()
	#np.matmul(np.matmul(np.array(sol['x']).T, H.numpy()), sol['x']) + np.matmul(f.numpy().T, np.array(sol['x']))
	return np.array(sol['x']), MMD_dist.item()
	#return solve_qp(H.numpy(), f.numpy(), A, b, None, None, lb, ub)
	


# for connected edges
def calc_feat_smooth(adj, features):
	A = sp.diags(adj.sum(1).flatten().tolist()[0])
	D = (A - adj)
	#(D * features) ** 2
	return (D * features)
	smooth_value = ((D * features) ** 2).sum() / (adj.sum() / 2 * features.shape[1])
	
	adj_rev = 1 - adj.todense()
	np.fill_diagonal(adj_rev, 0)

	A = sp.diags(adj_rev.sum(1).flatten().tolist()[0])
	D_rev = (A - adj_rev)
	smooth_rev_value = np.power(np.matmul(D_rev, features), 2).sum() / (adj_rev.sum() / 2 * features.shape[1])
	# D = torch.Tensor(D)
	
	return smooth_value, smooth_rev_value
	#return 

def calc_emb_smooth(adj, features):
	A = sp.diags(adj.sum(1).flatten().tolist()[0])
	D = (A - adj)
	return ((D * features) ** 2).sum() / (adj.sum() / 2 * features.shape[1])

def snowball(g, max_train, ori_idx_train, labels):
	train_seeds = set()

	label_cnt = defaultdict(int)
	train_ids = list(ori_idx_train)
	#random.shuffle(train_ids)
	# modify the snowball sampling into a function
	train_sampler = dgl.contrib.sampling.NeighborSampler(g, 1, -1,  # 0,
																neighbor_type='in', num_workers=1,
																add_self_loop=False,
																num_hops=2, seed_nodes=torch.LongTensor(train_ids), 
															   shuffle=True)
	cnt = 0
	for __, sample in enumerate(train_sampler):
		#option 1, 
		_center_label = labels[sample.layer_parent_nid(-1).tolist()[0]]
		if _center_label < 0:
			print('here')
			continue

		_center_id = sample.layer_parent_nid(-1).tolist()[0]
		#mbed()
		cnt += 1
		for i in range(sample.num_layers)[::-1][1:]:
			for idx in sample.layer_parent_nid(i).tolist():
				if idx == _center_id or labels[idx].item() < 0 or labels[idx].item() != _center_label.item():
					continue
				if idx not in train_seeds and label_cnt[labels[idx].item()] < max_train[labels[idx].item()] and idx in ori_idx_train:
					train_seeds.add(idx)
					label_cnt[labels[idx].item()] += 1
				
		#print(label_cnt)
		#if cnt == 5:
		#    break
		#print("iter", sample.layer_parent_nid(5))
		#init_labels = Counter(labels[list(train_seeds)])
		#if len(label_cnt.keys()) == num_class and min(label_cnt.values()) == max_train:
		done = True
		for k in range(labels.max().item()+1):
			if label_cnt[k] < max_train[k]:
				done = False
				break
		if done:
			break
	# print("number of seed used:{}".format(cnt))
	#print(label_cnt)
	return train_seeds, cnt
	# labels problem
def output_edgelist(g, OUT):
	for i,j in zip(g.edges()[0].tolist(), g.edges()[1].tolist()):
		OUT.write("{} {}\n".format(i, j))

def read_posit_emb(IN):
	tmp = IN.readline()
	a, b = tmp.strip().split(' ')
	emb = torch.zeros(int(a),int(b))
	for line in IN:
		tmp = line.strip().split(' ')
		emb[int(tmp[0]), :] = torch.FloatTensor(list(map(float, tmp[1:])))
	return emb

def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
	nnodes = adj_matrix.shape[0]
	A = adj_matrix + sp.eye(nnodes)
	D_vec = np.sum(A, axis=1).A1
	D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
	D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
	return D_invsqrt_corr @ A @ D_invsqrt_corr
	
def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
	nnodes = adj_matrix.shape[0]
	M = calc_A_hat(adj_matrix)
	A_inner = sp.eye(nnodes) - (1 - alpha) * M
	return alpha * np.linalg.inv(A_inner.toarray())
