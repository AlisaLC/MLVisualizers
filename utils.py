import io
import matplotlib.pyplot as plt
from PIL import Image

def fig2img():
    buf = io.BytesIO()
    plt.gcf().savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img