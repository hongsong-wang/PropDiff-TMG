import os
import numpy as np
from pathlib import Path
from scipy.ndimage import label, binary_fill_holes
from skimage.measure import marching_cubes, euler_number
import pandas as pd
import random
from tqdm import tqdm
from utils.mesh_utils import voxel2mesh
from sklearn.preprocessing import MinMaxScaler
import joblib

# torch.set_num_threads(2)

def compute_num_components(binary_voxel):
    labeled_array, num = label(binary_voxel, structure=np.ones((3, 3, 3)))
    return num

def compute_euler_number(binary_voxel):
    return euler_number(binary_voxel, connectivity=3)

def compute_surface_volume_ratio(binary_voxel, voxel_size=1.0):
    verts, faces, _, _ = marching_cubes(binary_voxel, level=0.5)
    def triangle_area(v0, v1, v2):
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    surface_area = sum(
        triangle_area(verts[i], verts[j], verts[k])
        for i, j, k in faces
    )
    volume = np.sum(binary_voxel) * voxel_size**3
    return surface_area / volume if volume > 0 else 0

def estimate_symmetry(binary_voxel):
    symmetric_x = np.all(binary_voxel == binary_voxel[::-1, :, :])
    symmetric_y = np.all(binary_voxel == binary_voxel[:, ::-1, :])
    symmetric_z = np.all(binary_voxel == binary_voxel[:, :, ::-1])
    if symmetric_x and symmetric_y and symmetric_z:
        return "cubic"
    elif symmetric_x or symmetric_y or symmetric_z:
        return "mirror"
    else:
        return "asymmetric"

def compute_has_voids(binary_voxel):
    binary_voxel = binary_voxel.astype(bool)
    filled = binary_fill_holes(binary_voxel)
    return np.any(filled & (~binary_voxel))

def extract_voxel_features(binary_voxel):
    return {
        "num_components": compute_num_components(binary_voxel),
        "euler_number": compute_euler_number(binary_voxel),
        "surface_volume_ratio": compute_surface_volume_ratio(binary_voxel),
        "symmetry": estimate_symmetry(binary_voxel),
        "has_voids": compute_has_voids(binary_voxel)
    }


# 2. 文本标注生成函数
def generate_structural_description(stiffness: dict, voxel_features: dict) -> str:
    C11_norm = stiffness.get("C11_norm", 0.0)
    C11 = stiffness.get("C11", 0.0)
    C12 = stiffness.get("C12", 0.0)
    C44_norm = stiffness.get("C44_norm", 0.0)
    C44 = stiffness.get("C44", 0.0)
    bulk_norm = stiffness.get("bulk_norm", 0.0)

    poisson_ratio = C12 / C11 if C11 > 1e-6 else 0.0

    description_parts = []
    intro = random.choice([
        "This mechanical metamaterial", 
        "The designed structure", 
        "This engineered microarchitecture"
    ])
    description_parts.append(intro)
    
    if bulk_norm > 1.0:
        bulk_texts = [
            "with exceptional resistance to volumetric compression",
            "featuring a highly stiff structure under uniform loading",
            "indicative of a strong, volume-preserving mechanical design"
        ]
    elif bulk_norm > -0.5:
        bulk_texts = [
            "with moderate resistance to volumetric deformation",
            "providing balanced compressive stiffness",
            "showing reasonable structural integrity under pressure"
        ]
    else:
        bulk_texts = [
            "but relatively compliant under bulk loading",
            "with low volumetric stiffness, prone to compression",
            "indicating a soft and easily compressible structure"
        ]

    description_parts.append(random.choice(bulk_texts))

    if C11_norm <= -1.5:
        axial_texts = ["is extremely soft and compliant under axial loads",
                   "yields easily with minimal resistance in the loading direction"]
    elif -1.5 < C11_norm <= -0.5:
        axial_texts = ["has low stiffness and flexes under pressure",
                    "shows compliant axial behavior"]
    elif -0.5 < C11_norm <= 0.5:
        axial_texts = ["demonstrates balanced axial stiffness",
                    "exhibits moderate rigidity under load"]
    elif 0.5 < C11_norm <= 1.5:
        axial_texts = ["maintains high stiffness along the principal axis",
                    "offers considerable resistance to axial deformation"]
    else:  # C11_norm > 1.5
        axial_texts = ["features extremely high stiffness in the loading direction",
                    "is ultra-rigid under axial compression"]
    description_parts.append(random.choice(axial_texts))

    # 泊松比
    if poisson_ratio < 0.0:
        poisson_texts = [
            "exhibits auxetic response with transverse widening upon stretching",
            "features counterintuitive deformation: expands sideways when pulled",
            "shows lateral dilation during tensile loading, indicative of auxeticity"
        ]
    if poisson_ratio < 0.05:
        poisson_texts = [
            "with near-zero lateral expansion indicating auxeticity",
            "showing minimal transverse strain typical of auxetics",
            "featuring decoupled lateral response as in auxetic materials"
        ]
    elif poisson_ratio < 0.3:
        poisson_texts = [
            "with low Poisson's ratio reducing lateral contraction",
            "showing partial auxetic behavior",
            "exhibiting mild auxetic tendencies"
        ]
    else:
        poisson_texts = [
            "and behaves like conventional materials under lateral loading",
            "with typical lateral contraction as seen in most solids",
            "showing standard transverse compression"
        ]
    description_parts.append(random.choice(poisson_texts))

    if C44_norm > 1.0:
        shear_texts = [
            "with excellent resistance to shear forces",
            "ideal for withstanding twisting and off-axis loads",
            "showcasing superior shear strength"
        ]
    elif C44_norm > -0.5:
        shear_texts = [
            "with moderate shear stiffness",
            "providing balanced resistance to torsion",
            "offering reasonable shear rigidity"
        ]
    else:
        shear_texts = [
            "but relatively weak under shear",
            "prone to deformation from twisting forces",
            "showing low resistance to transverse shear"
        ]
    description_parts.append(random.choice(shear_texts))

    # 连通性
    num_comp = voxel_features.get("num_components", 1)
    if num_comp == 1:
        connectivity_texts = [
            "The geometry forms a continuous single-component network",
            "It consists of one connected solid domain",
            "Structurally, it is fully connected without separation"
        ]
    else:
        connectivity_texts = [
            f"The structure is split into {num_comp} disconnected parts",
            f"It includes {num_comp} separate solid regions",
            f"There are {num_comp} distinct material segments present"
        ]
    description_parts.append(random.choice(connectivity_texts))

    # 空腔
    if voxel_features.get("has_voids", False):
        void_texts = [
            "with internal cavities and porous features",
            "containing voids or hollow sections within the bulk",
            "that reveal perforations or foam-like internal structure"
        ]
    else:
        void_texts = [
            "with a dense, solid internal matrix",
            "lacking voids or perforations",
            "and internally compact without cavities"
        ]
    description_parts.append(random.choice(void_texts))

    return ". ".join(description_parts) + "."

if __name__ == "__main__":
    scaler_E = joblib.load("scaler_C11")
    scaler_G = joblib.load("scaler_C44")
    scaler_v = joblib.load("scaler_C12")
    scaler_bulk = joblib.load("scaler_bulk")
    dataset_folder1 = "/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/rand8000"
    # dataset_folder2 = "/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/randombulk_compress"
    dataset_paths = []
    dataset_paths.extend(
        [p for p in Path(f'{dataset_folder1}').glob('**/*binary_voxel')])
    # dataset_paths.extend(
    #     [p for p in Path(f'{dataset_folder2}').glob('**/*binary_voxel')])
    records = []
    
    for path in tqdm(dataset_paths, desc="Processing voxel files"):
        with open(path, 'rb') as f:
            binary_data = np.unpackbits(np.fromfile(f, dtype=np.uint8))
            dim = (64, 64, 64)
            binary_array = np.reshape(binary_data, dim, order="F")
            binary_array = binary_array.astype(dtype=np.float32)
            fullvoxel = np.empty((128, 128, 128), dtype=np.float32, order='F')
            fullvoxel[64:128, 64:128, 64:128] = binary_array
            fullvoxel[0:64, 64:128, 64:128] = binary_array[::-1, :, :]
            fullvoxel[64:128, 64:128, 0:64] = binary_array[:, :, ::-1]
            fullvoxel[64:128, 0:64, 64:128] = binary_array[:, ::-1, :]
            fullvoxel[0:64, 0:64, 64:128] = binary_array[::-1, ::-1, :]
            fullvoxel[0:64, 64:128, 0:64] = binary_array[::-1, :, ::-1]
            fullvoxel[64:128, 0:64, 0:64] = binary_array[:, ::-1, ::-1]
            fullvoxel[0:64, 0:64, 0:64] = binary_array[::-1, ::-1, ::-1]
        print("+++++++++++++++++++++++++++++++++")
        mesh = voxel2mesh(fullvoxel)
        mesh.export("real.obj")
        tensor_path = str(path).replace("binary_voxel", "binary_C")
        with open(tensor_path, 'rb') as f:
            binary_data = np.fromfile(f, dtype=np.float32)
        
        # tensor_feature = - np.ones((10,), dtype=np.float32) 
        # tensor_feature[0] = binary_data[0]     # x方向弹性模量
        # tensor_feature[1] = binary_data[7]     # y方向弹性模量
        # tensor_feature[2] = binary_data[14]    # z方向弹性模量
        # tensor_feature[3] = binary_data[21]    # xy平面剪切模量
        # tensor_feature[4] = binary_data[28]    # yz平面剪切模量
        # tensor_feature[5] = binary_data[35]    # xz平面剪切模量
        # tensor_feature[6] = binary_data[1]     # 泊松比
        # tensor_feature[7] = binary_data[2]     # 其他耦合项
        # tensor_feature[8] = binary_data[8]     # 其他耦合项
        C11 = binary_data[0]
        C12 = binary_data[1]
        C44 = binary_data[21]
        bulk = (C11 + 2 * C12) / 3

        E_norm = scaler_E.transform(np.array([[C11]], dtype=np.float32))[0][0]
        v_norm = scaler_v.transform(np.array([[C12]], dtype=np.float32))[0][0]
        G_norm = scaler_G.transform(np.array([[C44]], dtype=np.float32))[0][0]
        bulk_norm = scaler_bulk.transform(np.array([[bulk]], dtype=np.float32))[0][0]

        stiffness = {
            "C11_norm": E_norm,
            "C11": C11,
            "C12": C12,
            "C44_norm": G_norm,
            "C44": C44,
            "bulk_norm": bulk_norm
        }
        features = extract_voxel_features(binary_array)
        description = generate_structural_description(stiffness, features)

        print("=== Generated Description ===")
        print(stiffness["C11_norm"], stiffness["C12"] / stiffness["C11"], stiffness["C44_norm"], stiffness["bulk_norm"])
        print(description)
        records.append({
            "file_name": os.path.basename(path),
            "E": stiffness["C11_norm"],
            "V": stiffness["C12"] / stiffness["C11"],
            "G": stiffness["C44_norm"],
            "bulk": stiffness["bulk_norm"],
            "num_components": features["num_components"],
            "has_voids": features["has_voids"],
            "description": description
        })

    df = pd.DataFrame(records)
    df.to_csv("/home/daibingxuan/workspace/microstructure_generation_3d/data/dataset1/voxel_descriptions_rand1.csv")
