# init the package folder
from dinf import dev_info

def gpuinfo(extended=False):
	''' Run the CUDA dev_info shared library to get info about the installed GPU devices. 
	'''

	if extended:
		info = dev_info(1)
		print info
	else:
		info = dev_info(0)

	

	return info
