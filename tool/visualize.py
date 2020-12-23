import imageio
import torch
import torchvision
from tool.dataloaders import create_batch_from_data_list


def generate_novel_views(model, img_source, azimuth_src, elevation_src,
                         azimuth_shifts, elevation_shifts):

    with torch.no_grad():
        num_views = len(azimuth_shifts)
        img_batch = img_source.unsqueeze(0)
        scenes = model.inverse_render(img_batch)
        scenes_batch = scenes.repeat(num_views, 1, 1, 1, 1)
        azimuth_src_batch = azimuth_src.repeat(num_views)
        elevation_src_batch = elevation_src.repeat(num_views)
        azimuth_target = azimuth_src_batch + azimuth_shifts
        elevation_target = elevation_src_batch + elevation_shifts
        rotated = model.rotate_source_to_target(scenes_batch, azimuth_src_batch,
                                                elevation_src_batch,
                                                azimuth_target, elevation_target)
    return model.render(rotated).detach()


def batch_generate_novel_views(model, imgs_source, azimuth_src,
                               elevation_src, azimuth_shifts,
                               elevation_shifts):
							   
    ##generates novel views for multiple images.
    num_imgs = imgs_source.shape[0]
    num_views = azimuth_shifts.shape[0]


    all_novel_views = [torch.zeros_like(imgs_source) for _ in range(num_views)]

    for i in range(num_imgs):
        # for each image
        novel_views = generate_novel_views(model, imgs_source[i],
                                           azimuth_src[i:i+1],
                                           elevation_src[i:i+1],
                                           azimuth_shifts, elevation_shifts).cpu()
        # add to each view.
        for j in range(num_views):
            all_novel_views[j][i] = novel_views[j]

    return all_novel_views


def dataset_novel_views(device, model, dataset, img_indices, azimuth_shifts,
                        elevation_shifts):

    # Extract image and pose information for all views
    data_list = []
    for img_idx in img_indices:
        data_list.append(dataset[img_idx])
    imgs_source, azimuth_src, elevation_src = create_batch_from_data_list(data_list)
    imgs_source = imgs_source.to(device)
    azimuth_src = azimuth_src.to(device)
    elevation_src = elevation_src.to(device)
    # Generate novel views
    return batch_generate_novel_views(model, imgs_source, azimuth_src,
                                      elevation_src, azimuth_shifts,
                                      elevation_shifts)



def save_generate_novel_views(filename, model, img_source, azimuth_src,
                              elevation_src, azimuth_shifts,
                              elevation_shifts):
    # generate novel views
    novel_views = generate_novel_views(model, img_source, azimuth_src,
                                       elevation_src, azimuth_shifts,
                                       elevation_shifts)
    # save raw image
    torchvision.utils.save_image(img_source, filename + '.png', padding=4,
                                 pad_value=1.)
    # save generated image
    torchvision.utils.save_image(novel_views, filename + '_novel.png',
                                 padding=4, pad_value=1.)

