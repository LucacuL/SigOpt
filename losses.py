import torch
import torch.nn as nn


#def VAE_Loss(X, res, h=None, **kwargs):
#	kl_scaling = .001
#	batch_dim = X.shape[0]
#	reconstr_error = nn.MSELoss()(X, res['X_hat'])
#	kl = (-0.5 * torch.sum(1 + (2*res['logstd']) - res['mean'].pow(2) - (2*res['logstd']).exp()))/batch_dim
#	if kwargs.get('train', False):
#		return {'full':reconstr_error + kl*kl_scaling*(kwargs['current_it'])/kwargs['total_it'], 'reconstruction':reconstr_error, 'kl':kl}
#	else:
#		return {'reconstruction':reconstr_error, 'kl':kl}
		
class VAE_Loss:
	def __init__(self, config):
		super().__init__()
		self.kl_scaling = config['kl_scaling']
		self.anneal = config['annealing']
		self.cycle = config['cycle']
		
	def __call__(self, X, res, **kwargs):
		batch_dim, emb_dim, _, _ = res['z'].shape
		reconstr_error = nn.MSELoss()(X, res['X_hat'])
		kl = (-0.5 * torch.sum(1 + (2*res['logstd']) - res['mean'].pow(2) - (2*res['logstd']).exp()))/(batch_dim*emb_dim)
		if len(kwargs)==0 or self.anneal is None:
			kl_scaling = self.kl_scaling
		elif self.anneal=='monotonic':
			kl_scaling = self.kl_scaling*kwargs['current_it']/kwargs['total_it']
		elif self.anneal=='cyclic':
            cycle_len = kwargs['total_it']/self.cycle
			anneal = kwargs['current_it']%(2*cycle_len)
			if anneal<=cycle_len:	
				anneal = anneal/cycle_len
			else:
				anneal = 1
			kl_scaling = self.kl_scaling*anneal
		else:
			kl_scaling = self.kl_scaling
		return {'full':reconstr_error + kl*kl_scaling, 'reconstruction':reconstr_error, 'kl':kl}


class Parallel_Loss:
	def __init__(self, config):
		super().__init__()
		self.alpha = config['loss_alpha']
		self.std = config['loss_noise']
	
	def __call__(self, X, res, h=None, **kwargs):
		reconstr_error = nn.MSELoss()(X, res['X_hat'])
		if h is None:
			inverse_loss = 0.
		else:
			h_temp = h + torch.randn_like(h).to(h.device)*self.std
			inverse_loss = nn.MSELoss()(h_temp, res['h_hat'])
		return {'full':reconstr_error + self.alpha*inverse_loss, 'reconstruction':reconstr_error, 'inverse_loss':inverse_loss}



#def Variational_Loss(X, res, h=None, **kwargs):
#	alpha = .01
#	batch_dim = X.shape[0]
#	reconstr_error = nn.MSELoss()(X, res['X_hat'])
#	kl = (-0.5 * torch.sum(1 + (2*res['logstd']) - res['mean'].pow(2) - (2*res['logstd']).exp()))/batch_dim	
#	if kwargs.get('train', False):
#		loss2 = nn.MSELoss()(h, res['h_hat'])
#		return {'full':reconstr_error + alpha*loss2, 'reconstruction':reconstr_error, 'kl':kl, 'strange':loss2}
#	else:
#		return {'reconstruction':reconstr_error, 'kl':kl}









