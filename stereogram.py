# Written by Simon Fuhrmann
# Some code from taken from the Automatic1111 plugin:
# https://github.com/thygate/stable-diffusion-webui-depthmap-script

from invokeai.invocation_api import (
    BaseInvocation,
    ColorField,
    ImageField,
    ImageOutput,
    InputField,
    InvocationContext,
    invocation,
)

from scipy import ndimage
from PIL import Image
import numpy as np

# Try to import numba's just-in-time (JIT) compilation which speeds up image
# warping by about 11x. If the import fails, provide a fallback for @njit.
try:
  from numba import njit
except Exception as e:
  print(f"Warning: numba import failed, stereo image generation will be slow")
  print(f"Warning: numba import error is: {str(e)}")
  def njit(*args, **kwargs):
    def decorator(func): return func
    return decorator


# This code treats rows of the image as polylines. It generates polylines,
# applies divergence to them, and then rasterizes them.
@njit(parallel=False)
def apply_divergence(color_img, depth_img, divergence_px, separation_px):
  EPSILON = 1e-7
  PIXEL_HALF_WIDTH = 0.0   # Sharp vs. smooth (use 0.45 for sharp)
  STEREO_OFFSET_EXP = 1.0  # Move objects more towards the far plane if >1

  h, w, c = color_img.shape
  derived_img = np.zeros_like(color_img)

  for row in range(h):
    depth_img_row = depth_img[row]
    color_img_row = color_img[row]
    derived_img_row = derived_img[row]

    # generating the vertices of the morphed polyline
    # format: new coordinate of the vertex, divergence (closeness),
    # column of pixel that contains the point's color
    pt = np.zeros((5 + 2 * w, 3), dtype=np.float_)
    pt_end: int = 0
    pt[pt_end] = [-1.0 * w, 0.0, 0.0]
    pt_end += 1
    for col in range(w):
      coord_d = (depth_img_row[col] ** STEREO_OFFSET_EXP) * divergence_px
      coord_x = col + 0.5 + coord_d + separation_px
      if PIXEL_HALF_WIDTH < EPSILON:
        pt[pt_end] = [coord_x, abs(coord_d), col]
        pt_end += 1
      else:
        pt[pt_end] = [coord_x - PIXEL_HALF_WIDTH, abs(coord_d), col]
        pt[pt_end + 1] = [coord_x + PIXEL_HALF_WIDTH, abs(coord_d), col]
        pt_end += 2
    pt[pt_end] = [2.0 * w, 0.0, w - 1]
    pt_end += 1

    # Generating the segments of the morphed polyline
    # Format: coord_x, coord_d, color_i of the first point, then the same for the second point
    sg_end: int = pt_end - 1
    sg = np.zeros((sg_end, 6), dtype=np.float_)
    for i in range(sg_end):
      sg[i] += np.concatenate((pt[i], pt[i + 1]))
    # Sort segments and points using insertion sort
    # Has a very good performance in practice, since these are almost sorted to begin with
    for i in range(1, sg_end):
      u = i - 1
      while pt[u][0] > pt[u + 1][0] and 0 <= u:
        pt[u], pt[u + 1] = np.copy(pt[u + 1]), np.copy(pt[u])
        sg[u], sg[u + 1] = np.copy(sg[u + 1]), np.copy(sg[u])
        u -= 1

    # Rasterizing
    # At each point in time we keep track of segments that are "active" (or "current")
    csg = np.zeros((5 * int(abs(divergence_px)) + 25, 6), dtype=np.float_)
    csg_end: int = 0
    sg_pointer: int = 0
    # and index of the point that should be processed next
    pt_i: int = 0
    for col in range(w):  # iterate over regions (that will be rasterized into pixels)
      color = np.full(c, 0.5, dtype=np.float_)  # we start with 0.5 because of how floats are converted to ints
      while pt[pt_i][0] < col:
        pt_i += 1
      pt_i -= 1  # pt_i now points to the dot before the region start
      # Finding segment' parts that contribute color to the region
      while pt[pt_i][0] < col + 1:
        coord_from = max(col, pt[pt_i][0]) + EPSILON
        coord_to = min(col + 1, pt[pt_i + 1][0]) - EPSILON
        significance = coord_to - coord_from
        # the color at center point is the same as the average of color of segment part
        coord_center = coord_from + 0.5 * significance

        # adding segments that now may contribute
        while sg_pointer < sg_end and sg[sg_pointer][0] < coord_center:
          csg[csg_end] = sg[sg_pointer]
          sg_pointer += 1
          csg_end += 1
        # removing segments that will no longer contribute
        csg_i = 0
        while csg_i < csg_end:
          if csg[csg_i][3] < coord_center:
            csg[csg_i] = csg[csg_end - 1]
            csg_end -= 1
          else:
            csg_i += 1
        # finding the closest segment (segment with most divergence)
        # note that this segment will be the closest from coord_from right up to coord_to, since there
        # no new segments "appearing" inbetween these two and _the polyline does not self-intersect_
        best_csg_i: int = 0
        if csg_end != 1:
          best_csg_closeness: float = -EPSILON
          for csg_i in range(csg_end):
            ip_k = (coord_center - csg[csg_i][0]) / (csg[csg_i][3] - csg[csg_i][0])
            # assert 0.0 <= ip_k <= 1.0
            closeness = (1.0 - ip_k) * csg[csg_i][1] + ip_k * csg[csg_i][4]
            if best_csg_closeness < closeness and 0.0 < ip_k < 1.0:
              best_csg_closeness = closeness
              best_csg_i = csg_i
        # getting the color
        col_l: int = int(csg[best_csg_i][2] + EPSILON)
        col_r: int = int(csg[best_csg_i][5] + EPSILON)
        if col_l == col_r:
          color += color_img_row[col_l] * significance
        else:
          ip_k = (coord_center - csg[best_csg_i][0]) / (csg[best_csg_i][3] - csg[best_csg_i][0])
          color += (color_img_row[col_l] * (1.0 - ip_k) +
                    color_img_row[col_r] * ip_k
                    ) * significance
        pt_i += 1
      derived_img_row[col] = np.asarray(color, dtype=np.uint8)
  return derived_img


@invocation("warp3d", title="Warp Image", tags=["image", "infill", "depth", "3d"], category="inpaint", version="1.0.0")
class WarpImage(BaseInvocation):
  """Warps an image using a depth map and infills the occlusions."""

  image: ImageField = InputField(description="Base image")
  depthmap: ImageField = InputField(description="A depth map to use")
  divergence: int = InputField(default=5, description="The intensity of the 3D effect")
  separation: int = InputField(default=0, description="Amount of image translation")


  def warp_image(self, color_img, depth_img, divergence, separation):
    # Convert depth map to grayscale and convert to numpy data.
    depth_img = depth_img.convert("L")
    depth = np.array(depth_img)

    # Convert the color image to RGBA.
    color_img = color_img.convert("RGBA")
    color = np.array(color_img)

    # Normalize the depth map.
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = (depth - depth_min) / (depth_max - depth_min)

    # Convert the divergence and separation values from percept to px.
    divergence_px = (divergence / 100.0) * color_img.width
    separation_px = (separation / 100.0) * color_img.width

    color = apply_divergence(color, normalized_depth, divergence_px, separation_px)

    return Image.fromarray(color)


  def invoke(self, context: InvocationContext) -> ImageOutput:
    img = context.images.get_pil(self.image.image_name)
    depthmap = context.images.get_pil(self.depthmap.image_name)
    warped = self.warp_image(img, depthmap, self.divergence, self.separation)
    warped_dto = context.images.save(image = warped)
    return ImageOutput.build(warped_dto)


@invocation("makestereopair", title="Make Stereo Pair", tags=["image", "infill", "depth", "3d"], category="inpaint", version="1.0.0")
class MakeStereoPair(BaseInvocation):
  """Horizontally stacks a left and right image including a border."""

  left: ImageField = InputField(description = "Left image")
  right: ImageField = InputField(description = "Right image")
  borderPx: int = InputField(default = 8, description = "Pixel size of the border")
  borderColor: ColorField = InputField(
    default = ColorField(r=255, g=255, b=255, a=255),
    description = "The color of the border",
  )


  def makePair(self, left, right, borderPx, borderColor):
    IMG_MODE = "RGBA"

    width = left.width + right.width + 3 * borderPx
    height = max(left.height, right.height) + 2 * borderPx
    pair = Image.new(IMG_MODE, (width, height))
    left.convert(IMG_MODE)
    right.convert(IMG_MODE)
    pair.paste(borderColor.tuple(), (0, 0, width, height))
    pair.paste(left, (borderPx, borderPx))
    pair.paste(right, (left.width + 2 * borderPx, borderPx))
    return pair


  def invoke(self, context: InvocationContext) -> ImageOutput:
    left = context.images.get_pil(self.left.image_name)
    right = context.images.get_pil(self.right.image_name)
    borderPx = max(0, self.borderPx)
    pair = self.makePair(left, right, borderPx, self.borderColor)

    pair_dto = context.images.save(image = pair)
    return ImageOutput.build(pair_dto)


@invocation("dilatedepth", title="Dilate Depth Map", tags=["image", "infill", "depth", "3d"], category="inpaint", version="1.0.0")
class DilateDepth(BaseInvocation):
  """Dilates a depth map."""

  depthmap: ImageField = InputField(description="The depth map to dilate")
  size: int = InputField(default = 1, description = "Half-size of the dilation stucturing element")


  def dilateDepth(self, depth):
    depth = depth.convert("L")
    depth_data = np.array(depth)
    depth_data = ndimage.grey_dilation(depth_data, self.size * 2 + 1)
    return Image.fromarray(depth_data)


  def invoke(self, context: InvocationContext) -> ImageOutput:
    depthmap = context.images.get_pil(self.depthmap.image_name)
    dilated = self.dilateDepth(depthmap)

    image_dto = context.images.save(image = dilated)
    return ImageOutput.build(image_dto)
