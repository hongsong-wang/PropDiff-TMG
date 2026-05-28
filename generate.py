import torch
import numpy as np
from network.model_trainer import DiffusionModel
from network.dual_encoder import TextEncoder
from utils.mesh_utils import voxel2mesh
from utils.utils import str2bool, ensure_directory
from utils.utils import num_to_groups
import argparse
import os
from tqdm import tqdm
import joblib
import math
import matplotlib.pyplot as plt
from pathlib import Path
import json
import random
import time
from bitstring import BitArray
import pdb

torch.set_num_threads(2)

def generate_based_on_text(
    model_path: str,
    query: str,
    prop: str,
    output_path: str = "./results",
    ema: bool = True,
    num_generate: int = 1,
    steps: int = 50,
    truncated_time: float = 0.0,
    w: float = 1.0,
):

    discrete_diffusion = DiffusionModel.load_from_checkpoint(model_path).cuda()
    postfix = f"text1"

    root_dir = os.path.join(output_path, postfix)
    ensure_directory(root_dir)

    
    query = [query]
        
    for ii in tqdm(range(0, len(query)), desc=f'save results in one batch in {root_dir}'):

        # tensor_c = np.array([E_data_map[0], G_data_map[0], v_data_map[0], vol[ii]], dtype=np.float32)
        # with open("/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/randombulk_compress/bulk_128_r_7238binary_voxel", 'rb') as f:
        #     voxel_data = np.unpackbits(np.fromfile(f, dtype=np.uint8))
        #     voxel = np.reshape(voxel_data, (64,64,64), order="F").astype(np.float32)

        # # 翻转修正
        # for i in range(8):
        #     tmp = voxel[i*8:(i+1)*8, :, :]
        #     voxel[i*8:(i+1)*8, :, :] = tmp[::-1]
        
        # mesh = voxel2mesh(voxel)
        # mesh.export("real.obj")
        
        
        generator = discrete_diffusion.ema_model if ema else discrete_diffusion.model
        time_start = time.time()
        res_tensor = generator.sample_with_text(caption=query[ii], prop=prop, batch_size=num_generate,
                                                    steps=steps, truncated_index=truncated_time, tensor_w=w)
        time_end = time.time()
        print('generte time cost', time_end - time_start, 's')
        voxel_res = []
        voxel_bt = []
        for jj in range(num_generate):
            voxel = res_tensor[jj].squeeze().cpu().numpy()
            voxel_res.append(voxel.copy())
            voxel[voxel>0] = 1
            voxel[voxel<0] = 0
            voxel_bt.append(voxel)
            # print(np.sum(voxel == 1))
            # print(voxel.shape)
            # np.save("exp_input_gen1.npy", voxel)
            # save to obj
            try:
                mesh = voxel2mesh(voxel)
                mesh.export(os.path.join(root_dir, str(ii)+'_'+str(jj) + ".obj"))
                fullvoxel = np.empty((128, 128, 128), dtype=np.float32, order='F')
                fullvoxel[64:128, 64:128, 64:128] = voxel
                fullvoxel[0:64, 64:128, 64:128] = voxel[::-1, :, :]
                fullvoxel[64:128, 64:128, 0:64] = voxel[:, :, ::-1]
                fullvoxel[64:128, 0:64, 64:128] = voxel[:, ::-1, :]
                fullvoxel[0:64, 0:64, 64:128] = voxel[::-1, ::-1, :]
                fullvoxel[0:64, 64:128, 0:64] = voxel[::-1, :, ::-1]
                fullvoxel[64:128, 0:64, 0:64] = voxel[:, ::-1, ::-1]
                fullvoxel[0:64, 0:64, 0:64] = voxel[::-1, ::-1, ::-1]
                meshfull = voxel2mesh(fullvoxel)
                meshfull.export(os.path.join(os.path.join(root_dir, str(ii)+'_'+str(jj) + ".obj")))
            except Exception as e:
                print(str(e))

            # ## save to voxel
            # flat_voxel = np.ravel(voxel)
            # bits_str = np.where(flat_voxel == 1, '0b1', '0b0')
            # bits = BitArray(''.join(bits_str))
            # reordered_bits = BitArray()
            # for i in range(0, len(bits), 8):
            #     byte = bits[i:i+8]
            #     reordered_bits.append(byte[::-1])  
            # out_filename = os.path.join(os.path.join(root_dir, str(ii)+'_'+str(jj) + ".voxel"))
            # with open(out_filename, 'wb') as f:
            #     reordered_bits.tofile(f)      
    return torch.tensor(voxel_res, dtype=torch.float32), torch.tensor(voxel_bt, dtype=torch.float32)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='generate something')
    parser.add_argument("--generate_method", type=str, default='generate_unconditional',
                        help="please choose :\n \
                            1. 'generate_unconditional' \n \
                            2. 'generate_based_on_tensor' \n \
                            3. 'latent_interpolation' \n \ ")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cls_model_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="text_results/")
    parser.add_argument("--input_path1", type=str, default="")
    parser.add_argument("--input_path2", type=str, default="")
    parser.add_argument("--ema", type=str2bool, default=True)
    parser.add_argument("--num_generate", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--truncated_time", type=float, default=0.0)
    parser.add_argument("--tensor_path", type=str, default="binary_C") 
    parser.add_argument('--query', nargs='+', default="The designed structure. with low volumetric stiffness, prone to compression. shows compliant axial behavior. with near-zero lateral expansion indicating auxeticity. but relatively weak under shear. that reveal perforations or foam-like internal structure.", metavar='N', help='text query array')
    parser.add_argument("--tensor_w", type=float, default=3.0)
    parser.add_argument("--cls", type=str2bool, default=False)
    parser.add_argument("--verbose", type=str2bool, default=False)

    args = parser.parse_args()
    method = (args.generate_method).lower()
    ensure_directory(args.output_path)

    if method == "generate_based_on_text":
        generate_based_on_text(model_path=args.model_path, output_path=args.output_path, ema=args.ema, steps=args.steps,
                                 num_generate=args.num_generate, truncated_time=args.truncated_time,
                                 query=args.query, prop=None, w=args.tensor_w)
    else:
        raise NotImplementedError
