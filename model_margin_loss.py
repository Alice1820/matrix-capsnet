# 2018.3.31

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal
import numpy as np

class PrimaryCaps(nn.Module):
	"""
	Args:
	A: input channel
	B: number of types of capsules
	"""
	def __init__(self, A=32, B=32):
		super(PrimaryCaps, self).__init__()
		self.B = B
		self.capsules_pose = nn.ModuleList([nn.Conv2d(in_channels=A, out_channels=4*4, kernel_size=1, stride=1)
											for _ in range(self.B)])
		self.capsules_activation = nn.ModuleList([nn.Conv2d(in_channels = A, out_channels = 1, kernel_size = 1, stride = 1)
											for _ in range(self.B)])

	def forward(self, x):
		poses = [self.capsules_pose[i](x) for i in range(self.B)]
		poses = torch.cat(poses, dim = 1)
		activations = [self.capsules_activation[i](x) for i in range(self.B)]
		activations = F.sigmoid(torch.cat(activations, dim = 1))
		return poses, activations

class ConvCaps(nn.Module):
	def __init__(self, batch_size, B=32, C=32, kernel=3, stride=2, iteration=3, coordinate_add=False, transform_share=False):
		super(ConvCaps, self).__init__()
		self.B = B
		self.C = C
		self.K = kernel  # kernel = 0 means full receptive field like class capsules
		self.Bkk = None
		self.Cww = None
		self.b = batch_size
		self.stride = stride
		self.coordinate_add = coordinate_add
		self.transform_share = transform_share
		self.beta_v = None
		self.beta_a = None
		if not transform_share:
			self.W = nn.Parameter(torch.randn(B, kernel, kernel, C, 4, 4))  # B,K,K,C,4,4
		else:
			self.W = nn.Parameter(torch.randn(B, C, 4, 4))  # B,C,4,4

		self.iteration = iteration

	def coordinate_addition(self, width_in, votes):
		add = [[i / width_in, j / width_in] for i in range(width_in) for j in range(width_in)]  # K,K,w,w
		add = Variable(torch.Tensor(add).cuda()).view(1, 1, self.K, self.K, 1, 1, 1, 2)
		add= add.expand(self.b, self.B, self.K, self.K, self.C, 1, 1, 2).contiguous()
		votes[:, :, :, :, :, :, :, :-2, -1] = votes[:, :, :, :, :, :, :, :-2, -1] + add
		return votes

	def down_w(self, w):
		return range(w*self.stride, w*self.stride+self.K)

	def EM_routing(self, lambda_, a_, V):
		'''
		Args:
		lambda_
		a_: activation function b,Bkk
		V: votes b,Bkk,Cww,4*4
		'''
		# routing coefficients
		R = Variable(torch.ones([self.b, self.Bkk, self.Cww]), requires_grad=False).cuda() / self.Cww # b,Bkk,Cww init to uniform distribution
		for i in range(self.iteration + 1):
			# M-step
			R = (R * a_)[...,None] # b,Bkk,Cww,1
			sum_R = R.sum(1) # b,Cww,1
			mu = ((R * V).sum(1) / sum_R)[:, None, :, :] # b,Cww,4*4
			sigma_square = (R * (V - mu) ** 2).sum(1) / sum_R # b,1,Cww,4*4

			# E-step
			if i != self.iteration:
				mu, sigma_square, V_, a__ = mu.data, sigma_square.data, V.data, a_.data
				normal= Normal(mu, sigma_square[:, None, :, :] ** (1 / 2)) # b,Cww,4*4
				p = torch.exp(normal.log_prob(V_)) # b,Bkk,Cww
				ap = a__ * p.sum(-1)
				R = Variable(ap / torch.sum(ap, -1)[..., None], requires_grad=False)
			else:
				cost = (self.beta_v.expand_as(sigma_square) + torch.log(sigma_square)) * sum_R # b,Cww,4*4
				a = torch.sigmoid(lambda_ * (self.beta_a.repeat(self.b, 1) - cost.sum(2))) # b,Cww
		return a, mu

	def forward(self, x, lambda_):
		'''
		Args:
		x: input, a tuple, (poses, activations)
			poses: [b,B,width_in,width_in,4*4]
			activations: [b,B,width_in,width_in]
		lambda_:
		Returns:
		_: output, a tuple, (poses, activations)
			poses: [b,C,w,w,4*4]
			activations: [b,C,w,w]
		'''
		poses, activations= x
		width_in = poses.size(2)
		w = int((width_in - self.K) / self.stride + 1) if self.K else 1 # width_out
		self.Cww = w * w * self.C
		self.b = poses.size(0)

		if self.beta_v is None:
			self.beta_v = nn.Parameter(torch.randn(1, self.Cww, 1)).cuda()
			self.beta_a = nn.Parameter(torch.randn(1, self.Cww)).cuda()

		if self.transform_share:
			if self.K  == 0:
				self.K = width_in
			W = self.W.view(self.B, 1, 1, self.C, 4, 4).expand(self.B, self.K, self.K, self.C, 4, 4).contiguous()
		else:
			W = self.W # B,K,K,C,4,4 parameters

		self.Bkk = self.K * self.K * self.B

		pose = poses.contiguous()
		pose = pose.view(self.b, 16, self.B, width_in, width_in).permute(0, 2, 3, 4, 1).contiguous()
		# print (poses)
		# poses = torch.stack([poses[:, :, self.stride*i:self.stride*i + self.K, self.stride*j:self.stride*j + self.K, :] for i in range(w) for j in range(w)], dim=-1) # b,B,K,K,w*w,16
		poses = torch.stack([pose[:, :, self.stride * i:self.stride * i + self.K, self.stride * j:self.stride * j + self.K, :] for i in range(w) for j in range(w)], dim=-1)  # b,B,K,K,w*w,16
		# print (poses)
		poses = poses.view(self.b, self.B, self.K, self.K, 1, w, w, 4, 4) # b,B,K,K,w,w,4,4
		W_hat = W[None, :, :, :, :, None, None, :, :] # 1,B,K,K,C,w,w,4,4
		votes = torch.matmul(W_hat, poses) # b,B,K,K,C,w,w,4,4

		if self.coordinate_add:
			votes = self.coordinate_addition(width_in, votes)
			activation = activations.view(self.b, -1)[..., None].repeat(1, 1, self.Cww)
		else:
			activations_ = [activations[:, :, self.down_w(x), :][:, :, :, self.down_w(y)] for x in range(w) for y in range(w)]
			activation = torch.stack(activations_, dim=4).view(self.b, self.Bkk, 1, -1) \
									.repeat(1, 1, self.C, 1).view(self.b, self.Bkk, self.Cww)

		votes = votes.view(self.b, self.Bkk, self.Cww, 16)
		activations, poses = getattr(self, 'EM_routing')(lambda_, activation, votes)
		return poses.view(self.b, self.C, w, w, -1), activations.view(self.b, self.C, w, w)

class CapsNet(nn.Module):
		def __init__(self, batch_size, img_size=28, A=32, B=32, C=32, D=32, E=10, r=3):
			super(CapsNet, self).__init__()
			self.num_classes = E
			self.conv1 = nn.Conv2d(in_channels=1, out_channels=A, kernel_size=5, stride=2)
			self.primary_caps = PrimaryCaps(A, B)
			self.convcaps1 = ConvCaps(batch_size, B, C, kernel=3, stride=2, iteration=r, coordinate_add=False, transform_share=False)
			self.convcaps2 = ConvCaps(batch_size, C, D, kernel=3, stride=1, iteration=r, coordinate_add=False, transform_share=False)
			self.classcaps = ConvCaps(batch_size, D, E, kernel=0, stride=1, iteration=r, coordinate_add=True, transform_share=True)
			self.decoder = nn.Sequential(nn.Linear(16 * E, 512),
										nn.ReLU(inplace=True),
										nn.Linear(512, 1024),
										nn.ReLU(inplace=True),
										nn.Linear(1024, img_size ** 2),
										nn.Sigmoid())

		def forward(self, x, lambda_, y=None):  # b,1,28,28
			x = F.relu(self.conv1(x))  # b,32,12,12
			x = self.primary_caps(x)  # b,32*(4*4+1),12,12
			x = self.convcaps1(x, lambda_)  # b,32*(4*4+1),5,5
			x = self.convcaps2(x, lambda_)  # b,32*(4*4+1),3,3
			p, a = self.classcaps(x, lambda_)  # b,10*16 b,10

			p = p.squeeze()

			# if y is None:
			_, y = a.max(dim=1)
			y = y.squeeze()

			# convert to one-hot
			y = Variable(torch.sparse.torch.eye(self.num_classes)).cuda().index_select(dim=0, index=y)

			reconstructions = self.decoder((p * y[:, :, None]).view(p.size(0), -1)) # b,28*28

			return a.squeeze(), reconstructions

class CapsuleLoss(nn.Module):
	def __init__(self, m, loss_type):
		super(CapsuleLoss, self).__init__()
		self.loss_type = loss_type
		self.reconstruction_loss = nn.MSELoss(size_average=False)
		self.multi_magin_loss = nn.MultiMarginLoss(p=2, margin=m)

	@staticmethod
	def spread_loss(x, target, m):
		loss = F.multi_margin_loss(x, target, p=2, margin=m)
		return loss

	@staticmethod
	def margin_loss(x, labels, m):
		left = F.relu(0.9 - x, inplace=True) ** 2
		right = F.relu(x - 0.1, inplace=True) ** 2

		labels = Variable(torch.sparse.torch.eye(x.size(1)).cuda()).index_select(dim=0, index=labels)

		margin_loss = labels * left + 0.5 * (1. - labels) * right
		margin_loss = margin_loss.sum()
		return margin_loss * 1/x.size(0)

	@staticmethod
	def cross_entropy_loss(x, target, m):
		loss = F.cross_entropy(x, target)
		return loss

	def forward(self, images, output, labels, m, recon):
		# main_loss = getattr(self, args.loss)(output, labels, m)
		# main_loss = getattr(self, 'spread_loss')(output, labels, m)
		main_loss = getattr(self, self.loss_type)(output, labels, m)
		# main_loss = self.multi_magin_loss(output, labels)
		# if args.use_recon:
		recon_loss = self.reconstruction_loss(recon, images)
		main_loss += 0.0005 * recon_loss

		return main_loss
