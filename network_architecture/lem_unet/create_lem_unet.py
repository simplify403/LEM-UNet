from nnunet_LEM.network_architecture.lem_unet.LEM_UNet import LEM_UNet


def create_lem_unet_medium(num_input_channels, num_classes, kernel_size=3, ds=False):

    return MedNeXt(
        in_channels = num_input_channels, 
        n_channels = 32,
        n_classes = num_classes, 
        exp_r=[2,3,4,4,4,4,4,3,2],       
        kernel_size=kernel_size,         
        deep_supervision=ds,             
        do_res=True,                     
        do_res_up_down = True,
        block_counts = [3,4,4,4,4,4,4,4,3],
        checkpoint_style = 'outside_block'
    )




def create_lem_unet(num_input_channels, num_classes, model_id, kernel_size=3,
                      deep_supervision=False):

    model_dict = {
        'M': create_lem_unet_medium,
        }
    
    return model_dict[model_id](
        num_input_channels, num_classes, kernel_size, deep_supervision
        )


if __name__ == "__main__":

    model = create_lem_unet_medium(1, 3, 3, False)
    print(model)