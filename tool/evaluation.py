import torch
import torch.nn.functional as F



def create_data_batch(data_list):
    imgs = []
    azimuths = []
    elevations = []
    for data_item in data_list:
        img, render_params = data_item["img"], data_item["render_params"]
        azimuth, elevation = render_params["azimuth"], render_params["elevation"]
        imgs.append(img.unsqueeze(0))
        azimuths.append(torch.Tensor([azimuth]))
        elevations.append(torch.Tensor([elevation]))
    imgs = torch.cat(imgs, dim=0)
    azimuths = torch.cat(azimuths)
    elevations = torch.cat(elevations)
    return imgs, azimuths, elevations



def get_dataset_psnr(device, model, dataset, source_img_idx_shift=64, batch_size=16):
    #Compute PSNR for each scene in the test dataset 
	
	
    num_imgs_per_scene = dataset.num_imgs_per_scene
    num_scenes = dataset.num_scenes


    assert (num_imgs_per_scene - 1) % batch_size == 0
    batches_per_scene = (num_imgs_per_scene - 1) // batch_size

    mean_psnrs = []
    for i in range(num_scenes):
        # extract the 64th image in source view
        source_img_idx = i * num_imgs_per_scene + source_img_idx_shift
        img_source = dataset[source_img_idx]["img"].unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
        render_params = dataset[source_img_idx]["render_params"]
        azimuth_source = torch.Tensor([render_params["azimuth"]]).repeat(batch_size).to(device)
        elevation_source = torch.Tensor([render_params["elevation"]]).repeat(batch_size).to(device)
        
		# Infer 3D source view
        scenes = model.inverse_render(img_source)  ##3D scene


        num_points_in_batch = 0
        data_list = []
        psnr = 0.
        for j in range(num_imgs_per_scene):
            if j == source_img_idx_shift:
                continue 

            data_list.append(dataset[i * num_imgs_per_scene + j])
            num_points_in_batch += 1
   
            if num_points_in_batch == batch_size:
                # Create batch for gt data
                img_gt, azimuth_target, elevation_target = create_data_batch(data_list)
                img_gt = img_gt.to(device)
                azimuth_target = azimuth_target.to(device)
                elevation_target = elevation_target.to(device)
				
                # rotate and generate prediction.
                rotated = model.rotate_source_to_target(scenes, azimuth_source,
                                                        elevation_source, azimuth_target,
                                                        elevation_target)
                img_predicted = model.render(rotated).detach()
				# Compute psnr
                psnr += get_psnr(img_predicted, img_gt)
                data_list = []
                num_points_in_batch = 0

        mean_psnrs.append(psnr / batches_per_scene)

        print("{}/{}: Current - {:.3f}, Mean - {:.4f}".format(i + 1,
                                                              num_scenes,
                                                              mean_psnrs[-1],
                                                              torch.mean(torch.Tensor(mean_psnrs))))

    return mean_psnrs


def get_psnr(prediction, gt):

    batch_size = prediction.shape[0]
    mse_all_pixel = F.mse_loss(prediction, gt, reduction='none')
    mse_per_img = mse_all_pixel.view(batch_size, -1).mean(dim=1)
    psnr = 10 * torch.log10(1 / mse_per_img)
	mean_psnr = torch.mean(psnr).item()
    return mean_psnr
	
	

