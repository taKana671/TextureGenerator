# TextureGenerator

Procedually generate texture images from noise.
The `noise` of this repository contains two directories, `cynoise` and `pynoise`. `cynoise` has Cython source code and must be built by setup.py to use.
If it has been build, the fast Cython code is used, if not, python code is used.  
See [NoiseTexture](https://github.com/taKana671/NoiseTexture/tree/main) for the difference in speed.

# Requirements

* Cython 3.0.12
* numpy 2.2.4
* opencv-contrib-python 4.11.0.86
* opencv-python 4.11.0.86

# Environment

* Python 3.12
* Windows11

# Building Cython code

### Clone this repository with submodule.
```
git clone --recursive https://github.com/taKana671/TextureGenerator.git
```

### Build cython code.
```
cd TextureGenerator
python setup.py build_ext --inplace
```
If the error like "ModuleNotFoundError: No module named ‘distutils’" occurs, install the setuptools.
```
pip install setuptools
```

# Procedural texture
Output procedural texture images from noise.  
For more details of methods and parameters, please see source codes.

1. [Cloud](#cloud)
2. [Cubemap](#cubemap)
3. [Sphere map](#sphere-map)
4. [Island heightmap](#island-heightmap)

## Cloud

![Image](https://github.com/user-attachments/assets/017ab598-c65c-4a76-9819-470cd78ca941)

```
from texture_generator.cloud import Cloud

maker = Cloud.from_vfractal()             # using fractal Value Noise.

# maker = Cloud.from_pfractal()           # using fractal Perlin Noise.
# maker = Cloud.from_sfractal()           # using fractal Simplex Noise.
# maker = Cloud.from_file('noise_5.png')  # using noise image file.

# output a cloud image composited with a background image. 
maker.create_cloud_image()
```


## Cubemap

The cubemap is seamless and each skybox image can be connected exactly.  
See [skybox](https://github.com/taKana671/skybox).

![Image](https://github.com/user-attachments/assets/a27a2d3c-4dcd-4275-b952-b5691695d0f2)

```
from texture_generator.cubemap import CubeMap
from utils.output_image import output

maker = CubeMap.from_sfractal()           # using fractal Simplex Noise.

# maker = CubeMap.from_pfractal()         # using fractal Perlin Noise.
# maker = CubeMap.from_vfractal()         # using fractal Value Noise.

# output a grayscale cubemap image.
img = maker.create_cubemap()
output(img, 'cubemap')

# output skybox images composited with a background image.
maker.create_skybox_images()
```

## Sphere map

![Image](https://github.com/user-attachments/assets/6de22ecc-8759-4fee-b9dc-5759e5c29729)

```
from texture_generator.spheremap import SphereMap
from utils.output_image import output

# using fractal Simplex Noise.
maker = SphereMap.from_sfractal()
 
# output a grayscale sphere map image.
img = maker.create_spheremap()
output(img, 'spheremap')

# output a sky sphere image composited with a background image.    
maker.create_skysphere_image()
```

![Image](https://github.com/user-attachments/assets/5dd59dc1-7c0b-45f4-a804-fccaa6cabe6d)

```
from texture_generator.spheremap import SphereMap
from output_image import output

# using fractal Simplex Noise.
maker = SphereMap.from_sfractal()

# output a grayscale cubemap image converted from a sphere map.
img = maker.create_cubemap()
output(img, 'cubemap')

# output skybox images composited with a background image.
maker.create_skybox_images()
```

## Island heightmap

![Image](https://github.com/user-attachments/assets/74f43288-9f68-4a77-a608-23a0b1daba4b)

```
from heightmap_generator.island_heightmap import IslandHeightmap

maker = IslandHeightmap.from_sfractal()
# maker = IslandHeightmap.from_sfractal(reverse=True)  # basin-shaped heightmap
maker.create_island_heightmap()
```
