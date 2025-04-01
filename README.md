# texture_generator

A submodule repository for [NoiseTexture](https://github.com/taKana671/NoiseTexture)  
Procedually generate texture images from noise.

# Procedural texture
Output procedural texture images from noise.  
For more details of methods and parameters, please see source codes.

1. [Cloud](#cloud)
2. [Skybox](#skybox)

## Cloud
![Image](https://github.com/user-attachments/assets/017ab598-c65c-4a76-9819-470cd78ca941)

```
from texture_generator.cloud import Cloud

maker = Cloud.from_vfractal()             # using fractal Value Noise.
# maker = Cloud.from_pfractal()           # using fractal Perlin Noise.
# maker = Cloud.from_sfractal()           # using fractal Simplex Noise.
# maker = Cloud.from_file('noise_5.png')  # using noise image file.
maker.create_cloud_image()
```


## Skybox
The cubemap is completely seamless and each skybox image can be connected exactly.  
See [skybox](https://github.com/taKana671/skybox).

![Image](https://github.com/user-attachments/assets/a27a2d3c-4dcd-4275-b952-b5691695d0f2)

```
from texture_generator.cubemap import CubeMap

maker = CubeMap.from_sfractal()           # using fractal Simplex Noise.
# maker = CubeMap.from_pfractal()         # using fractal Perlin Noise.
maker.create_skybox_images()
```
