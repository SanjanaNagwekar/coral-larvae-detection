from __future__ import annotations
import argparse
from pathlib import Path
import cv2
from src.detection import TIMEPOINT0,TIMEPOINT1,detect_coral_larvae,profile_for_filename
from src.edge_detection import canny_edges,sobel_edges
from src.preprocessing import gaussian_blur,load_image
from src.texture_segmentation import texture_segmentation

IMAGE_EXTENSIONS={".jpg",".jpeg",".png"}

def parse_args()->argparse.Namespace:
    parser=argparse.ArgumentParser(description="Run the cleaned coral-larvae computer-vision pipeline.")
    parser.add_argument("--input",required=True,type=Path,help="Path to one image or a directory of images.")
    parser.add_argument("--output",type=Path,default=Path("outputs"),help="Directory for generated pipeline stages.")
    parser.add_argument("--profile",choices=("auto","timepoint0","timepoint1"),default="auto")
    parser.add_argument("--with-texture",action="store_true",help="Also run Law's texture-energy + K-Means segmentation.")
    return parser.parse_args()

def image_paths(path:Path)->list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []
    if path.is_dir():
        return sorted(i for i in path.iterdir() if i.is_file() and i.suffix.lower() in IMAGE_EXTENSIONS)
    return []

def choose_profile(name:str,filename:str):
    if name=="timepoint0": return TIMEPOINT0
    if name=="timepoint1": return TIMEPOINT1
    return profile_for_filename(filename)

def save_image(path:Path,image)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),image):
        raise RuntimeError(f"Failed to write image: {path}")

def process_image(source:Path, output_root:Path, profile_name:str, with_texture:bool)->int:
    image=load_image(source)
    blurred=gaussian_blur(image)
    profile=choose_profile(profile_name,source.name)
    detected,mask,count=detect_coral_larvae(image,profile)
    dest=output_root/source.stem
    save_image(dest/"01_gaussian.jpg",blurred)
    save_image(dest/"02_sobel.jpg",sobel_edges(blurred))
    save_image(dest/"03_canny.jpg",canny_edges(blurred))
    save_image(dest/"04_region_mask.jpg",mask)
    save_image(dest/"05_detections.jpg",detected)
    if with_texture:
        save_image(dest/"06_texture_segmentation.jpg",texture_segmentation(image))
    print(f"{source.name}: detected {count} circular candidate region(s)")
    return count

def main()->None:
    args=parse_args()
    sources=image_paths(args.input)
    if not sources:
        raise SystemExit(f"No supported images found at: {args.input}")
    for source in sources:
        process_image(source,args.output,args.profile,args.with_texture)

if __name__=="__main__":
    main()
