import numpy as np
import random
import cv2
from noise import SimplexNoise, PerlinNoise, Fractal2D
from noise.output_image import output_image_8bit
from mask.radial_gradient_generator import RadialGradientMask
from utils import output_image


# public float[,] GenerateNoise(int mapSize,int octaves, string seed, float noiseScale, float persistence, float lacunarity, Vector2 offset)
#     {
#         if (noiseScale <= 0)
#         {
#             noiseScale = 0.0001f;
#         }
        
#         float halfWidth = mapSize / 2f;
#         float halfHeight = mapSize / 2f;
        
#         float[,] noiseMap = new float[mapSize + 1, mapSize + 1];
#         System.Random rand = new System.Random(seed.GetHashCode());

#         //Octaves offset
#         Vector2[] octavesOffset = new Vector2[octaves];
#         for (int i = 0; i < octaves; i++)
#         {
#             float offset_X = rand.Next(-100000, 100000) + offset.x;
#             float offset_Y = rand.Next(-100000, 100000) + offset.y;
#             octavesOffset[i] = new Vector2(offset_X / mapSize, offset_Y / mapSize);
#         }

#         for (int x = 0; x < mapSize; x++)
#         {
#             for (int y = 0; y < mapSize; y++)
#             {
#                 float amplitude = 1;
#                 float frequency = 1;
#                 float noiseHeight = 0;
#                 float superpositionCompensation = 0;


#                 for (int i = 0; i < octaves; i++)
#                 {
#                     float sampleX = (x - halfWidth) / noiseScale * frequency + octavesOffset[i].x;
#                     float sampleY = (y - halfHeight) / noiseScale * frequency + octavesOffset[i].y;

#                     float perlinValue = Mathf.PerlinNoise(sampleX, sampleY);
#                     noiseHeight += perlinValue * amplitude;
#                     noiseHeight -= superpositionCompensation;

#                     amplitude *= persistence;
#                     frequency *= lacunarity;
#                     superpositionCompensation = amplitude / 2;

#                 }

#                 noiseMap[x, y] = Mathf.Clamp01(noiseHeight);

#             }
#         }

#         return noiseMap;
    # }

# persistence = 0.375  # 0.5
# lacunarity = 2.52    # 2.5


def clamp(x, min_val, max_val):
        """Args:
            x (float): the value to constrain.
            min_val (float): the lower end of the range into which to constrain x.
            max_val (float): the upper end of the range into which to constrain x.
        """
        return min(max(x, min_val), max_val)


def inverse_lerp(a, b, v):
    ret = 0.

    if a != b:
        ret = (v - a) / (b - a)

    ret = 1. if ret > 1. else ret
    ret = 0. if ret < 0. else ret 
    return ret


def sample(size=256, octaves=3, scale=200, persistence=0.375, lacunarity=2.52):
    half_w = size / 2
    half_h = size / 2
    noise = PerlinNoise()
    offset = [0, 0]
    arr = []
    # arr = np.zeros(size + 1, size + 1)
    # octaves_offset = np.zeros(octaves)
    li = []

    for i in range(octaves):
        offset_x = random.uniform(-100000, 100000) + offset[0]
        offset_y = random.uniform(-100000, 100000) + offset[1]
        li.append([offset_x / size, offset_y / size])

    octaves_offset = np.array(li)

    for x in range(size):
        for y in range(size):
            amplitude = 1.
            frequency = 1.
            noise_height = 0.
            superpos_compensation = 0.

            for i in range(octaves):
                sample_x = (x - half_w) / scale * frequency + octaves_offset[i][0]
                sample_y = (y - half_h) / scale * frequency + octaves_offset[i][1]

                v = noise.pnoise2(sample_x, sample_y)
                noise_height += v * amplitude
                noise_height -= superpos_compensation

                amplitude *= persistence
                frequency *= lacunarity
                superpos_compensation = amplitude / 2

            arr.append(clamp(v, 0, 1))

    arr = np.array(arr)
    return arr
    # arr = arr.reshape(size, size)
    # output_image_8bit(arr)


def generate_falloff_map(size=256):
    li = []

    for x in range(size):
        for y in range(size):
            # idx = x * size + y
            falloff_a = x / size * 2
            falloff_b = y / size * 2
           
            v = max(abs(falloff_a), abs(falloff_b))
            ev = evaluate(v)
            rv = radial_fall_off(ev, 100, x, y, size / 2., size / 2.)
            # rv = feathered_radial_fall_off(v, 200, 400, x, y, size / 2., size / 2.)

            li.append(rv)

    arr = np.array(li)
    return arr


def feathered_radial_fall_off(v, inner_radius, outer_radius, x, y, cx, cy):
    dx = cx - x
    dy = cy - y
    dist_sqr = dx ** 2 + dy ** 2
    irad_sqr = inner_radius ** 2
    orad_sqr = outer_radius ** 2

    if dist_sqr >= orad_sqr:
        return 0.

    if dist_sqr <= irad_sqr:
        return v

    dist = dist_sqr ** 0.5
    t = inverse_lerp(inner_radius, outer_radius, dist)

    return v * t



def evaluate(v):
    a = 3.
    b = 2.2

    return v ** a / (v ** a + (b - b * v) ** a)


def radial_fall_off(v, radius, x, y, cx, cy):
    dx = x - cx
    dy = y - cy
    dist_sqr = (dx ** 2 + dy ** 2)
    rad_sqr = radius ** 2
    # print(dist_sqr)
    if dist_sqr > rad_sqr:
        return 0.
    return v


def main(size=256):
    arr = sample()
    # import pdb; pdb.set_trace()
    # output_image_8bit(arr.reshape(size, size))
    offset = generate_falloff_map()
    # output_image_8bit(offset.reshape(size, size))
    tmp = arr - offset
    
    img = tmp.reshape(size, size)

    output_image_8bit(img)


def circular_gradient(size=256, radius=50):
    arr = np.zeros([256, 256, 3], dtype=np.uint8)
    inner_color = (0, 0, 0)
    outer_color = (255, 255, 255)
    center = (size / 2, size / 2)
    center = (100, 100)

    for y in range(size):
        for x in range(size):
            dist_to_center = ((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5
            dist = dist_to_center / (size / 2 * 2 ** 0.5)
            # dist = dist_to_center / 50

            if dist >= 1:
                r, g, b = outer_color
            else:
                r = outer_color[0] * dist + inner_color[0] * (1 - dist)
                g = outer_color[1] * dist + inner_color[1] * (1 - dist)
                b = outer_color[2] * dist + inner_color[2] * (1 - dist)

            # print(y, x, dist, [r, g, b])
            arr[y, x] = [r, g, b]
    
    cv2.imwrite("red_sample_mask_3.png", arr)


def mask_img():
    # img = cv2.imread("img8_20250824101213.png")
    img = cv2.imread("img8_20250824105858.png") # perlin fractal
    # img = cv2.imread("img8_20250824110201.png")
    # mask = cv2.imread("sample_mask.png")
    mask = cv2.imread("radial_gradient_20250905234904.png")

    h, w = img.shape[:2]

    for y in range(h):
        for x in range(w):
            img_v = img[y, x, 0]
            mask_v = mask[y, x, 0]
            
            if mask_v >= img_v:
                v = 0
            else:
                # v = img_v - (img_v - mask_v)
                # v = img[y, x, 0]
                v = img_v - mask_v
                # v = int(img_v * (mask_v / 255))
            # if (v := img_v - mask_v) <= 0:
                # v = 0
            img[y, x] = v

    cv2.imwrite("new_img_130.png", img)


def mask_from_value(height=129, width=129):
    mask = RadialGradientMask(height=height, width=width)
    simplex = SimplexNoise()
    noise = Fractal2D(simplex.snoise2)
    t = simplex.mock_time()

    noise_arr = np.zeros((height, width))
    mask_arr = np.zeros((height, width)) 

    arr = np.zeros((height, width))
    m = min(height, width)

    for y in range(height):
        for x in range(width):
            nv = noise.fractal(x / m + t, y / m + t)
            mv = mask.get_gradient(x, y)[0]
            # import pdb; pdb.set_trace()
            v = 0 if mv >= nv else nv - mv
            arr[y, x] = v
            noise_arr[y, x] = nv
            mask_arr[y, x] = mv

    arr = np.clip(arr * 255, a_min=0, a_max=255).astype(np.uint8)
    output_image.output(arr, 'island')

    arr = np.clip(noise_arr * 255, a_min=0, a_max=255).astype(np.uint8)
    output_image.output(arr, 'noise')

    arr = np.clip(mask_arr * 255, a_min=0, a_max=255).astype(np.uint8)
    output_image.output(arr, 'mask')


# if __name__ == '__main__':
#     # sample()
#     # main()

#     # circular_gradient()
#     # mask_img()
#     mask_from_value()




