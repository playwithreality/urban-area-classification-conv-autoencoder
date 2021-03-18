from PIL import Image
from os import listdir

base_dir = "./openSAR/patch_Calib"
folders = listdir(base_dir)
out_dir = "./openSAR/png_calib/"

im = Image.open("openSAR/patch_Calib/airport/airport_x1437_y14390_Polygon6_Calib.tif")
print("GOT IMG")

for folder in folders:
    files = listdir(base_dir+"/"+folder)
    for file in files:
        im = Image.open(base_dir+"/"+folder+"/"+file)
        print("xxx")
        im.thumbnail(im.size)
        im.save(out_dir+folder+"/"+file[:-3], "PNG")
