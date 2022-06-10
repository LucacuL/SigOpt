import time
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

from utils import update_dict, mean_dict



class Model(ABC):
	def __init__(self, config : Dict):
		super().__init__()
		self.config = config
		self.model = config['model'].to(self.config['device'])
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])#, weight_decay= 1.e-04)
		self.scheduler = StepLR(self.optimizer, step_size=config['scheduler_step'], gamma=0.7)
	
	def _save(self, extend_name=''):
		try:
			state_dict = self.model.module.state_dict()
		except AttributeError:
			state_dict = self.model.state_dict()
		torch.save(state_dict, './models/'+self.config['name']+extend_name+'.pt')
		print('Model saved')	
	
	@staticmethod	
	def _plot(plots : Dict, x_axis : Optional[List] = None, name : str = 'plot', save : bool = True):	
		for p in plots.keys():
			n_plots = len(plots[p].keys())
			fig, axs = plt.subplots(nrows=1, ncols=n_plots, figsize=(30,10))
			for i, key in enumerate(plots[p].keys()):
				if x_axis is None or len(x_axis)!=len(plots[p][key]):
					xs = np.arange(len(plots[p][key]))
				else:
					xs = x_axis
				axs[i].plot(xs, plots[p][key], label=key)
				axs[i].legend()
			plt.tight_layout()
			if save:
				fig.savefig(name+'_'+p+'.png')
			else:
				plt.show()
			plt.close()
			
	@abstractmethod
	def fit(self):
		return None
	
	def test(self, data : Union[DataLoader, Dataset], losses_log : Optional[Dict] = None):
	    if isinstance(data, Dataset):
	        dataloader = DataLoader(data, batch_size=self.config['batch'], shuffle=False)
	    else:
	        dataloader = data
		self.model.eval()
		flag = False
		if losses_log is None:
			flag = True
			losses_log = {}
		with torch.no_grad():
			cum_losses = {}
			for data, _ in dataloader:
				X = data.to(self.config['device'])
				res = self.model(X)
				losses = self.config['loss'](X, res)
				update_dict(cum_losses, losses)
			mean_dict(cum_losses)
			update_dict(losses_log, cum_losses)
			if flag:
				return losses_log
			else:
				print(f"\t Val loss: {cum_losses}")



class ParallelAE(Model):		
	def fit(self, train_data : Dataset, val_data : Optional[Dataset] = None, plot : bool = False):
		train_loader = DataLoader(train_data, batch_size=self.config['batch'], shuffle=True)
		if val_data is not None:
			val_loader = DataLoader(val_data, batch_size=self.config['batch'], shuffle=False)
		losses_log = {'train' : {}, 'val' : {}}
		time_steps = []
		best_val = (np.inf, 0)
		start = time.time()
		for epoch in range(1, self.config['epochs']+1):
			self.model.train()
			print(f'-Epoch {epoch}:')
			cum_losses = {}
			for data, _ in train_loader:
				if self.config['parallel']:
#					h = torch.rand(data.shape[0], self.model.emb_dim, 1, 1).to(self.config['device'])*2. - 1.
					h = (torch.randn(data.shape[0], self.model.emb_dim, 1, 1)*self.config['lat_radius']).to(self.config['device'])
				else:
					h = None
				X = data.to(self.config['device'])
				self.optimizer.zero_grad()
				res = self.model(X, h=h)
				losses = self.config['loss'](X, res, h=h)
				update_dict(cum_losses, losses)
				if self.config['parallel'] and epoch<=self.config['warmup']:
					loss = losses['inverse_loss']
				else:
					loss = losses['full']
				loss.backward()
				self.optimizer.step()
			if self.optimizer.param_groups[0]['lr'] > 1e-5:
				self.scheduler.step()
			if self.optimizer.param_groups[0]['lr'] < 1e-5:
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = 1e-4
			mean_dict(cum_losses)
			update_dict(losses_log['train'], cum_losses)
			print(f"\t Train loss: {cum_losses}")
			if val_data is not None:
				self.test(val_loader, losses_log['val'])
				if losses_log['val']['reconstruction'][-1] < best_val[0]:
					best_val = losses_log['val']['reconstruction'][-1], epoch+1
					self._save()
			time_steps.append((time.time()-start)/60.)
#			if time_steps[-1]>self.config['time_lim']:
#				print(time_steps[-1])
#				break
		print(f'best validation error : {best_val[0]} at epoch {best_val[1]}')
		self._save('_last')
		if plot:
			self._plot(losses_log, name = self.config['name'], x_axis=time_steps)	
		return losses_log	
	
	
				
				
				
class Variational(Model):	
	def fit(self, train_data : Dataset, val_data : Optional[Dataset] = None, plot : bool = False):
		train_loader = DataLoader(train_data, batch_size=self.config['batch'], shuffle=True)
		if val_data is not None:
			val_loader = DataLoader(val_data, batch_size=self.config['batch'], shuffle=False)
		
		losses_log = {'train' : {}, 'val' : {}}
		time_steps = []
		best_val = (np.inf, 0)
		start = time.time()
		total_iterations = len(train_loader)*(self.config['epochs']-self.config['warmup'])
		counter = 0
		for epoch in range(1, self.config['epochs']+1):
			self.model.train()
			print(f'-Epoch {epoch}:')
			cum_losses = {}
			for data, _ in train_loader:
				X = data.to(self.config['device'])
				self.optimizer.zero_grad()
				res = self.model(X)
				losses = self.config['loss'](X, res, total_it=total_iterations, current_it=counter)
				update_dict(cum_losses, losses)
				if epoch<=self.config['warmup']:
					loss = losses['reconstruction']
				else:
					loss = losses['full']	
				loss.backward()
				self.optimizer.step()
				counter+=1
			if self.optimizer.param_groups[0]['lr'] > 1e-5:
				self.scheduler.step()
			if self.optimizer.param_groups[0]['lr'] < 1e-5:
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = 1e-4
			mean_dict(cum_losses)
			update_dict(losses_log['train'], cum_losses)
			print(f"\t Train loss: {cum_losses}")
			if val_data is not None:
				self.test(val_loader, losses_log['val'])
				if losses_log['val']['reconstruction'][-1] < best_val[0]:
					best_val = losses_log['val']['reconstruction'][-1], epoch+1
					self._save()
			time_steps.append((time.time()-start)/60.)
#			if time_steps[-1]>self.config['time_lim']:
#				print(time_steps[-1])
#				break
		print(f'best validation error : {best_val[0]} at epoch {best_val[1]}')
		self._save('_last')
		if plot:
			self._plot(losses_log, name = self.config['name'], x_axis=time_steps)	
		return losses_log	
				
				
				
				
				
	
