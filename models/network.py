import torch
import torch.nn as nn
from torch.nn import init
import functools
import pdb
import sys
sys.dont_write_bytecode = True

''' 
	
	This Network is designed for Few-Shot Learning Problem. 

'''


###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('Linear') != -1:
		init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('Linear') != -1:
		init.xavier_normal_(m.weight.data, gain=0.02)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
	classname = m.__class__.__name__
	# print(classname)
	if classname.find('Conv') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
	classname = m.__class__.__name__
	print(classname)
	if classname.find('Conv') != -1:
		init.orthogonal_(m.weight.data, gain=1)
	elif classname.find('Linear') != -1:
		init.orthogonal_(m.weight.data, gain=1)
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
	print('initialization method [%s]' % init_type)
	if init_type == 'normal':
		net.apply(weights_init_normal)
	elif init_type == 'xavier':
		net.apply(weights_init_xavier)
	elif init_type == 'kaiming':
		net.apply(weights_init_kaiming)
	elif init_type == 'orthogonal':
		net.apply(weights_init_orthogonal)
	else:
		raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
	elif norm_type == 'none':
		norm_layer = None
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer



def define_CovarianceNet(pretrained=False, model_root=None, which_model='Conv64', norm='batch', init_type='normal', use_gpu=True, **kwargs):
	CovarianceNet = None
	norm_layer = get_norm_layer(norm_type=norm)

	if use_gpu:
		assert(torch.cuda.is_available())

	if which_model == 'Conv64':
		CovarianceNet = CovarianceNet_64(norm_layer=norm_layer, **kwargs)
	else:
		raise NotImplementedError('Model name [%s] is not recognized' % which_model)
	init_weights(CovarianceNet, init_type=init_type)

	if use_gpu:
		CovarianceNet.cuda()

	if pretrained:
		CovarianceNet.load_state_dict(model_root)

	return CovarianceNet


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)



##############################################################################
# Classes: CovarianceNet_64
##############################################################################

# Model: CovarianceNet_64 : the embedding backbone has 4 convolutional layers
# Input: One query image and a support set
# Architecture: 4 Convolutional layers --> Covariance metric layer --> Classification layer  
# Dataset: 84 x 84 x 3, for miniImageNet, StanfordDog, StanfordCar, CubBird
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21


class CovarianceNet_64(nn.Module):
	def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5):
		super(CovarianceNet_64, self).__init__()

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		self.features = nn.Sequential(                       # 3*84*84
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),           # 64*42*42

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),
			nn.MaxPool2d(kernel_size=2, stride=2),           # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),                         # 64*21*21

			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
			norm_layer(64),
			nn.LeakyReLU(0.2, True),                         # 64*21*21
		)
		
		self.covariance = CovaBlock()                        # 1*(441*num_classes)

		self.classifier = nn.Sequential(
			nn.LeakyReLU(0.2, True),
			nn.Dropout(),
			nn.Conv1d(1, 1, kernel_size=441, stride=441, bias=use_bias),
		)


	def forward(self, input1, input2):

		# extract features of input1--query image
		q = self.features(input1)

		# extract features of input2--support set
		S = []
		for i in range(len(input2)):
			S.append(self.features(input2[i]))

		x = self.covariance(q, S) # get Batch*1*(h*w*num_classes)
		x = self.classifier(x)    # get Batch*1*num_classes
		x = x.squeeze(1)          # get Batch*num_classes

		return x



#========================== Define a Covariance Metric layer ==========================#
# Calculate the local covariance matrix of each category in the support set
# Calculate the Covariance Metric between a query sample and a category


class CovaBlock(nn.Module):
	def __init__(self):
		super(CovaBlock, self).__init__()


	# calculate the covariance matrix 
	def cal_covariance(self, input):
		
		CovaMatrix_list = []
		for i in range(len(input)):
			support_set_sam = input[i]
			B, C, h, w = support_set_sam.size()

			support_set_sam = support_set_sam.permute(1, 0, 2, 3)
			support_set_sam = support_set_sam.contiguous().view(C, -1)
			mean_support = torch.mean(support_set_sam, 1, True)
			support_set_sam = support_set_sam-mean_support

			covariance_matrix = support_set_sam@torch.transpose(support_set_sam, 0, 1)
			covariance_matrix = torch.div(covariance_matrix, h*w*B-1)
			CovaMatrix_list.append(covariance_matrix)

		return CovaMatrix_list    


	# calculate the similarity  
	def cal_similarity(self, input, CovaMatrix_list):
	
		B, C, h, w = input.size()
		Cova_Sim = []
	
		for i in range(B):
			query_sam = input[i]
			query_sam = query_sam.view(C, -1)
			query_sam_norm = torch.norm(query_sam, 2, 1, True)    
			query_sam = query_sam/query_sam_norm

			if torch.cuda.is_available():
				mea_sim = torch.zeros(1, len(CovaMatrix_list)*h*w).cuda()

			for j in range(len(CovaMatrix_list)):
				temp_dis = torch.transpose(query_sam, 0, 1)@CovaMatrix_list[j]@query_sam
				mea_sim[0, j*h*w:(j+1)*h*w] = temp_dis.diag()

			Cova_Sim.append(mea_sim.unsqueeze(0))

		Cova_Sim = torch.cat(Cova_Sim, 0) # get Batch*1*(h*w*num_classes)
		return Cova_Sim 


	def forward(self, x1, x2):

		CovaMatrix_list = self.cal_covariance(x2)
		Cova_Sim = self.cal_similarity(x1, CovaMatrix_list)

		return Cova_Sim
