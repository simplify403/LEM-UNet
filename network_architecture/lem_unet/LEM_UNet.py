import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from nnunet_LEM.network_architecture.lem_unet.blocks import *


class LEM_UNet(nn.Module):

    def __init__(self, 
        in_channels: int, 
        n_channels: int,
        n_classes: int, 
        exp_r: int = 4,                            # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,                      # Ofcourse can test kernel_size
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,             # Can be used to test deep supervision
        do_res: bool = False,                       # Can be used to individually test residual connection
        do_res_up_down: bool = False,             # Additional 'res' connection on up and down convs
        checkpoint_style: bool = None,            # Either inside block or outside block
        block_counts: list = [2,2,2,2,2,2,2,2,2], # Can be used to test staging ratio: 
                                            # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
        norm_type = 'group',
        dim = '3d',                                # 2d or 3d
        grn = False
    ):

        super().__init__()

        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        assert dim in ['2d', '3d']
        
        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
            
        self.stem = conv(in_channels, n_channels, kernel_size=1)
        self.stem_edge = conv(1, n_channels, kernel_size=1)
        if type(exp_r) == int:
            exp_r = [exp_r for i in range(len(block_counts))]
        
        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                ) 
            for i in range(block_counts[0])]
        ) 

        self.down_0 = MedNeXtDownBlock(
            in_channels=n_channels,
            out_channels=2*n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim
        )
    
        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
                exp_r=exp_r[1],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[1])]
        )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2*n_channels,
            out_channels=4*n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
                exp_r=exp_r[2],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[2])]
        )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4*n_channels,
            out_channels=8*n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )
        
        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*8,
                out_channels=n_channels*8,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )            
            for i in range(block_counts[3])]
        )
        
        self.down_3 = MedNeXtDownBlock(
            in_channels=8*n_channels,
            out_channels=16*n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*16,
                out_channels=n_channels*16,
                exp_r=exp_r[4],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[4])]
        )

        self.up_3 = MedNeXtUpBlock(
            in_channels=16*n_channels,
            out_channels=8*n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*8,
                out_channels=n_channels*8,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[5])]
        )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8*n_channels,
            out_channels=4*n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[6])]
        )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4*n_channels,
            out_channels=2*n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[7])]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2*n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[8])]
        )

        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)

        # Used to fix PyTorch checkpointing bug
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)  

        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels*2, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=n_channels*4, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=n_channels*8, n_classes=n_classes, dim=dim)
            self.out_4 = OutBlock(in_channels=n_channels*16, n_classes=n_classes, dim=dim)

        self.block_counts = block_counts

        self.cbam1 = CBAM_Block(32)
        self.cbam2 = CBAM_Block(64)

        self.expand0 = nn.Sequential(nn.Conv3d(33, 32, kernel_size=1),
                                     nn.BatchNorm3d(32),
                                     nn.GELU())
        self.expand1 = nn.Sequential(nn.Conv3d(65, 64, kernel_size=1),
                                     nn.BatchNorm3d(64),
                                     nn.GELU())
        self.c_change = nn.Sequential(nn.Conv3d(33, 65, kernel_size=1),
                                      nn.BatchNorm3d(65),
                                      nn.GELU())
        self.down_EA = nn.Sequential(
                                    nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # 二倍下采样
                                    nn.BatchNorm3d(64),
                                    nn.GELU()
                                )
        # self.c_change_1 = nn.Conv3d(64, 65, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.enc_block_edge_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=2,
                kernel_size=3,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(3)]
                                         )
        self.down_edge = MedNeXtDownBlock(
            in_channels=n_channels,
            out_channels=n_channels*2,
            exp_r=3,
            kernel_size=3,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim
        )

        self.enc_block_edge_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
                exp_r=2,
                kernel_size=3,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for i in range(3)])
        self.scale0 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.scale1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.norm0 = nn.BatchNorm3d(32)
        self.norm1 = nn.BatchNorm3d(64)


 
 
    def iterative_checkpoint(self, sequential_block, x):

        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x


    def forward(self, x):
        x_edge = x
        tensor_list = torch.split(x, split_size_or_sections=1, dim=1)
        t1_mri = tensor_list[0]
        t1_attention = self.stem_edge(t1_mri)
        x = self.stem(x)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor)#64
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor)#128
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor)#256
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor)#512

            attention_0_main = self.cbam1(x_res_0)
            attention_1_main = self.cbam2(x_res_1)
            pyr_list = make_laplace_pyramid(x_edge, 2)##3层大小分别是128，64，32
            edge_0 = pyr_list[0]
            edge_1 = pyr_list[1]
            edge_encoder_0 = self.enc_block_edge_0(t1_attention)
            attention0 = self.cbam1(edge_encoder_0)
            edge_down = self.down_edge(edge_encoder_0)
            edge_encoder_1 = self.enc_block_edge_1(edge_down)
            attention1 = self.cbam2(edge_encoder_1)


            sum0 = self.norm0(self.scale0*attention_0_main + (1-self.scale0)*attention0)
            sum1 = self.norm1(self.scale1*attention_1_main + (1-self.scale1)*attention1)
            

            fusion1 = torch.cat((edge_0, sum0), dim=1)
            fusion2 = torch.cat((edge_1, sum1), dim=1)
            # fusion1_1 = self.c_change(fusion1) 
            fusion1_1 = F.interpolate(fusion1, scale_factor=0.5, mode='trilinear')
            fusion1_1 = self.c_change(fusion1_1)
            # x_res_1_1 = self.c_change_1(x_res_1)
            expend0 = self.expand0(fusion1)
            expend1 = self.expand1(fusion2)
            expend0_use = self.down_EA(expend0)
            expend1_use = self.sigmoid(expend1)
            mutil = expend0_use * expend1_use + expend1
            


            x = self.iterative_checkpoint(self.bottleneck, x)
            if self.do_ds:
                x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)

            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_3 
            x = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x, self.dummy_tensor)
            del x_res_3, x_up_3

            x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor)
            dec_x = x_res_2 + x_up_2 
            x = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x, self.dummy_tensor)
            del x_res_2, x_up_2

            x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor)
            x_up_1_1 = make_laplace(x_up_1)
            feature1 = self.sigmoid(x_up_1_1)
            dec_x = x_res_1 + x_up_1 + mutil*feature1
            x = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x, self.dummy_tensor)
            del x_res_1, x_up_1

            x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor)
            x_up_0_0 = make_laplace(x_up_0)
            feature0 = self.sigmoid(x_up_0_0)
            dec_x = x_res_0 + x_up_0 + feature0*expend0
            x = self.iterative_checkpoint(self.dec_block_0, dec_x)
            del x_res_0, x_up_0, dec_x

            x = checkpoint.checkpoint(self.out_0, x, self.dummy_tensor)

        else:
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)

            x = self.bottleneck(x)
            if self.do_ds:
                x_ds_4 = self.out_4(x)

            x_up_3 = self.up_3(x)
            dec_x = x_res_3 + x_up_3 
            x = self.dec_block_3(dec_x)

            if self.do_ds:
                x_ds_3 = self.out_3(x)
            del x_res_3, x_up_3

            x_up_2 = self.up_2(x)
            dec_x = x_res_2 + x_up_2 
            x = self.dec_block_2(dec_x)
            if self.do_ds:
                x_ds_2 = self.out_2(x)
            del x_res_2, x_up_2

            x_up_1 = self.up_1(x)
            dec_x = x_res_1 + x_up_1 
            x = self.dec_block_1(dec_x)
            if self.do_ds:
                x_ds_1 = self.out_1(x)
            del x_res_1, x_up_1

            x_up_0 = self.up_0(x)
            dec_x = x_res_0 + x_up_0 
            x = self.dec_block_0(dec_x)
            del x_res_0, x_up_0, dec_x

            x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else: 
            return x


if __name__ == "__main__":

    network = LEM_UNet(
            in_channels = 3, 
            n_channels = 32,
            n_classes = 2,
            exp_r=[2,3,4,4,4,4,4,3,2],        # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [3,4,4,4,4,4,4,4,3],
            checkpoint_style = 'outside_block',
            dim = '3d'
            
        ).cuda()

    from thop import profile
    from thop import clever_format
    x = torch.rand((1, 3, 128, 128, 128),requires_grad=False).cuda()
    
    flops, params = profile(network, inputs=(x, ))
    # print(flops, params)
    macs, params = clever_format([flops, params], "%.3f")
    print(macs,params)
    import time
    start_time = time.time()
    for i in range(100):
        output  = network(x)
        del output
    end_time = time.time()
    print('time:', (end_time - start_time)/100)
    # print(network)