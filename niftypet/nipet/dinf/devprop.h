
struct PropGPU{
	char	n_gpu;		//number of all GPUs
	char	*name;		//GPU name
	int 	totmem;		//total available memory
	int		cc_major;	//compute capability major and minor
	int 	cc_minor;
};


PropGPU *devprop(char showprop);

void copy_string(char d[], char s[]);