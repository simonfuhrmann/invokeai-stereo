# Stereogram Custom Nodes for InvokeAI

> [!NOTE]
> The `main` branch is for InvokeAI 4.x. Please use the `invoke3` branch if
> you're on an older version of InvokeAI.

This repository contains custom nodes for InvokeAI to create cross-view or
parallel-view stereograms. Stereograms are 2D images that, when viewed properly,
reveal a 3D scene. Check out
[r/crossview](https://www.reddit.com/r/CrossView/wiki/index/) for tutorials.

Following is an example stereo image with an embedded workflow. To load the
workflow, drag and drop it into your InvokeAI (copy and Ctrl-V seems to drop
metdata, including the workflow). Then right-click the image and select "Load
Workflow".

![Example stereo image](docs/example_stereo_01.png)

## Installation

Clone this repository into your `<invokeai>/nodes/` directory. Then restart
InvokeAI. This will make new nodes available in the workflow editor (see below
for details). For example:

```bash
cd $INVOKE_ROOT/nodes
git clone https://github.com/simonfuhrmann/invokeai-stereo.git
```

Alternatively, create a directory under your `<invokeai>/nodes/` directory, and
copy `stereogram.py` and `__init__.py` into this directory.

Alternatively (NOT recommended), the `stereogram.py` can also be placed into the
application directory under
`<invokeai>/.venv/lib/python3.10/site-packages/invokeai/app/invocations/`.

## Custom nodes

The file `stereogram.py` contains three nodes for InvokeAI's workflow system. An
example cross-view workflow for the nodes is here. To turn this into a
parallel-view image, simply swap the left/right images on the "Make Stereo
Pair" node.

![Example workflow screenshot](docs/example_workflow_01.png)

In particular, these nodes are provided:

* **Dilate Depth Map**: This node is optional. It takes an input depth map and
  performs a dialtion operation on the depth map. This causes close depth values
  (white) to expand spatially. This is useful for depth maps that are
  auto-generated by AI models (such as DepthAnything), which are usually quite
  tight around the foreground objects, leading to foreground colors bleeding
  into the background during

* **Warp Image**: Takes an input depth map and color image, and warps the color
  image according to geometric information from the depth map, and fills
  occlusions. The node can be configured using a `divergence` and a `separation`
  value (both in percept).
  Larger divergence causes a stronger 3D effect (and more artifacts). Separation
  simply shifts the image.
  A positive divergence generates right-eye images, a negative divergence
  generates left-eye images. (But note for cross-view, the images have to be
  swapped.) A positive separation value causes the image to get shifted left, a
  negative separation value causes the image to get shifted right.

* **Make Stereo Pair**: Horizontally stacks a left and right input image into
  a stereogram. Optionally, a border is drawn around the left and right image
  with a configurable color.

## Cross-view vs. parallel-view

* Cross-view stereograms are viewed by crossing your eyes such that the left
  image pane becomes the right-eye image, and the right image pane becomes the
  left eye image.
* Parallel-view stereograms are viewed by diverging your eyes such that left
  and right images are viewed by the left and right eye, respectively. Note, for
  larger disparities (e.g., on a computer screen), cross-view images are much
  easier to view.

## Technical details

Image warping converts per-pixel depth values into disparities. Pixels in the
input image are then shifted to their new location in the output image with
respect to these disparities. This reveals occluded (unknown) areas in the
output image that need to be filled.

Filling occlusions is a hard problem, and this implementation is by no means
perfect. Most of the warping implementation has been taken from the
[A1111 plugin here](https://github.com/thygate/stable-diffusion-webui-depthmap-script).
