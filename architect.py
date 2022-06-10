import torch
import torch.nn as nn



#'k' :[2, 3, 3, 4, 2], 's': [1, 2, 2, 2, 1]
#'k' :[3, 5, 3, 5, 3], 's': [1, 2, 1, 2, 1]; 'k' :[3, 5, 3, 6, 3], 's': [1, 2, 1, 2, 1]	

class PlainAE(nn.Module):
	def __init__(self, in_channels = 1, emb_dim = 8):
		super().__init__()
		self.emb_dim = emb_dim
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=0),  
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.Conv2d(8, 8, kernel_size=5, stride=2, padding=0), 
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0), 
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=0),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.Conv2d(16, emb_dim, kernel_size=3, stride=1, padding=0), 
#			nn.ReLU(True), 
#			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(emb_dim, 16, kernel_size=3, stride=1, padding=0),  
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=0),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=0), 
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.ConvTranspose2d(8, 8, kernel_size=6, stride=2, padding=0),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.ConvTranspose2d(8, in_channels, kernel_size=3, stride=1, padding=0), 
			)
			
	def encode(self, X):
		return self.encoder(X)

	def forward(self, X, *args):
		h = self.encode(X)
		X_hat = self.decoder(h)
		return {'emb':h, 'X_hat':X_hat}


class AE(PlainAE):
	def __init__(self, in_channels = 1, emb_dim = 8):
		super().__init__(in_channels=in_channels, emb_dim=emb_dim)

	def forward(self, X, h=None):
		emb = self.encoder(X)
		X_hat = self.decoder(emb)
		if h is None:
			img = None
			h_hat = None 
		else:
			img = self.decoder(h)
			h_hat = self.encoder(img)
		return {'emb':emb, 'X_hat':X_hat, 'img':img, 'h_hat':h_hat}
		
		
			
class VAE(nn.Module):
	def __init__(self, in_channels = 1, emb_dim = 2):
		super().__init__()
		self.emb_dim = emb_dim
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=0),  
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.Conv2d(8, 8, kernel_size=5, stride=2, padding=0), 
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0), 
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=0),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			)
		self.conv_mean = nn.Sequential(
			nn.Conv2d(16, emb_dim, kernel_size=3, stride=1, padding=0), 
			)
		self.conv_std = nn.Sequential(
			nn.Conv2d(16, emb_dim, kernel_size=3, stride=1, padding=0), 
			nn.ReLU(True),
			)
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(emb_dim, 16, kernel_size=3, stride=1, padding=0),  
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=0),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=0), 
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.ConvTranspose2d(8, 8, kernel_size=6, stride=2, padding=0),
			nn.LeakyReLU(negative_slope=0.1, inplace=True),
			nn.ConvTranspose2d(8, in_channels, kernel_size=3, stride=1, padding=0), 
			)
		
	def encode(self, X):
		batch_dim = X.shape[0]
		X = self.encoder(X)
		mean = self.conv_mean(X)
		logstd = self.conv_std(X)
		return mean, logstd
		
	def reparameterize(self, mean, logstd):
		std = torch.exp(logstd)
		eps = torch.randn_like(std).to(std.device)
		return mean + eps*std

	def forward(self, X, *args):
		mean, logstd = self.encode(X)
		z = self.reparameterize(mean, logstd)
		X_hat = self.decoder(z)
		return {'mean': mean, 'logstd':logstd, 'z':z, 'X_hat':X_hat}	
		
		
		
		
		
		
		
		
		
		
		
		
		
		
