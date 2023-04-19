import gradio as gr

from colorizers import *

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

title = "Colorize Images"
description = """
Colorize black and white images using the ECCV 2016 and SIGGRAPH 2017 colorization papers by Zhang et al.:
- Colorful Image Colorization: https://arxiv.org/abs/1603.08511
- Real-Time User-Guided Image Colorization with Learned Deep Priors: https://arxiv.org/abs/1705.02999
<br>
Reference implementation: https://github.com/richzhang/colorization
<br>
Adapted to Gradio by DIGIMAP Group 12:
- GREGORIO, DALE PONS LEE
- SILLONA, JOHN EUGENE JUSTINIANO
- SY, MATTHEW JERICHO GO
"""


def color(image, ver):
    # default size to process images is 256x256
    # grab L channel in both original ("orig") and resized ("rs") resolutions
    (tens_l_orig, tens_l_rs) = preprocess_img(image, HW=(256, 256))

    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel
    if ver == "eccv16":
        out_img = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    else:
        out_img = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    return out_img


gr.Interface(
    fn=color,
    inputs=[
        "image",
        gr.Radio(
            ["eccv16", "siggraph17"], type="value", value="eccv16", label="version"
        ),
    ],
    # outputs=[gr.Image(label="eccv16"), gr.Image(label="siggraph17")],
    outputs="image",
    allow_flagging="never",
    title=title,
    description=description,
    examples=[
        ["imgs/moon-captured-bw-lg.jpeg", "eccv16"],
    ],
).launch()
