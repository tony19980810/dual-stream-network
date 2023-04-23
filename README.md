# dual-stream-network
Masked facial expression recognition based on temporal overlap model and AU-GCN

## Citation
If you find this work useful for your research,please cite our paper:

## Method
For details, please refer to the above paper. The follwing briefly summarize our work here.
- 1.a combined CNNs and transformer sequence recognition model that can learn from scratch on MFED.
- 2.temporal overlap model which enlarges the temporal receptive field of each token and enables better combination of temporal features.
- 3.GCN with AU (action unit) intensity used as node features and a 3D learnable adjacency matrix based on AU activation states. 
- 4.A  dual-stream network which combines the features of image stream (transformer) and AU stream (GCN).

## Required Package:
	tqdm 4.64.1
	torchvision 0.10.1+cu111
	torchaudio 0.9.1
	scipy 1.7.1
	pytorch 1.9.1
	python 3.8.11
	pillow 8.3.1
	opencv 4.0.1
	numpy 1.18.5
	
## File instructions:
	caculate_all_result.py:  Calculate overall acc,F1,recall using txt result files of every subject in LOSO in txt folder.
	generate_adj_for_MFED.py: Generate node matrix, 3D learnable adjacency matrix and mask matrix.
	generate_txt.py: Generate train and test dataset using LOSO.
	MEIP_Micro_FACS_Codes.jason:Record dataset labels and other related information.
	my_dataset.py:Custom dataset code.
	utils.py: Include code for training, testing, and calculating metrics.
	model:The specific code of the model.
	explain.py,rollout.py,grad_rollout.py:Code for visualization.
