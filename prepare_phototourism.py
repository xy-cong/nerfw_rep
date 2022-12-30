import argparse
# from lib.datasets import PhototourismDataset
from lib.datasets.nerfw.new_photo import PhototourismDataset
import numpy as np
import os
import pickle
from lib.config import cfg,args
 
if __name__ == '__main__':
    # args = get_opts()
    print(cfg.train_dataset.img_downscale)
    os.makedirs(os.path.join(args.data_root, 'cache'), exist_ok=True)
    print(f'Preparing cache for scale {cfg.train_dataset.img_downscale}...')
    nerfw_arg = cfg.train_dataset 
    print(nerfw_arg)
    dataset = PhototourismDataset(**nerfw_arg)
    # save img ids
    with open(os.path.join(args.data_root, f'cache/img_ids.pkl'), 'wb') as f:
        pickle.dump(dataset.img_ids, f, pickle.HIGHEST_PROTOCOL)
    ######################### new #########################
    with open(os.path.join(args.data_root, f'cache/img_to_cam_id.pkl'), 'wb') as f:
        pickle.dump(dataset.image_to_cam, f, pickle.HIGHEST_PROTOCOL)
    ######################### new #########################
    # save img paths
    with open(os.path.join(args.data_root, f'cache/image_paths.pkl'), 'wb') as f:
        pickle.dump(dataset.image_paths, f, pickle.HIGHEST_PROTOCOL)
    # save Ks
    with open(os.path.join(args.data_root, f'cache/Ks{cfg.train_dataset.img_downscale}.pkl'), 'wb') as f:
        pickle.dump(dataset.Ks, f, pickle.HIGHEST_PROTOCOL)
    # save scene points
    np.save(os.path.join(args.data_root, 'cache/xyz_world.npy'),
            dataset.xyz_world)
    # save poses
    np.save(os.path.join(args.data_root, 'cache/poses.npy'),
            dataset.poses)
    # save near and far bounds
    with open(os.path.join(args.data_root, f'cache/nears.pkl'), 'wb') as f:
        pickle.dump(dataset.nears, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.data_root, f'cache/fars.pkl'), 'wb') as f:
        pickle.dump(dataset.fars, f, pickle.HIGHEST_PROTOCOL)
    # save rays and rgbs
    np.save(os.path.join(args.data_root, f'cache/rays{cfg.train_dataset.img_downscale}.npy'),
            dataset.all_rays.numpy())
    np.save(os.path.join(args.data_root, f'cache/rgbs{cfg.train_dataset.img_downscale}.npy'),
            dataset.all_rgbs.numpy())
    print(f"Data cache saved to {os.path.join(args.data_root, 'cache')} !")