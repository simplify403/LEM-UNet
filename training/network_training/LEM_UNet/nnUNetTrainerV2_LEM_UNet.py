import torch
import os
import torch.nn as nn
from nnunet_LEM.network_architecture.lem_unet.LEM_UNet import LEM_UNet as LEM_UNet_Orig
from nnunet_LEM.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet_LEM.network_architecture.neural_network import SegmentationNetwork
from nnunet_LEM.utilities.nd_softmax import softmax_helper


class LEM_UNet(LEM_UNet_Orig, SegmentationNetwork):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 2**5
        self.num_classes = kwargs['n_classes']
        # self.do_ds = False        Already added this in the main class


class nnUNetTrainerV2_Optim_and_LR(nnUNetTrainerV2):

    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-3

    def process_plans(self, plans):
        super().process_plans(plans)
        # Please don't do this for nnunet. This is only for MedNeXt for all the DS to be used
        num_of_outputs_in_mednext = 5
        self.net_num_pool_op_kernel_sizes = [[2,2,2] for i in range(num_of_outputs_in_mednext+1)]    
    
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), 
                                            self.initial_lr, 
                                            weight_decay=self.weight_decay,
                                            eps=1e-4        # 1e-8 might cause nans in fp16
                                        )
        self.lr_scheduler = None





class nnUNetTrainerV2_LEM_UNet_M_kernel3(nnUNetTrainerV2_Optim_and_LR):   
        
    def initialize_network(self):
        self.network = LEM_UNet(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [3,4,4,4,4,4,4,4,3],
            checkpoint_style = 'outside_block'
        )

        if torch.cuda.is_available():
            self.network.cuda()



