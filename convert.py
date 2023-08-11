#optimum-cli export onnx  --model saved_pix2struct_model --task image-to-text-with-past google/pix2struct-docvqa-base

import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, required=True)
argparser.add_argument('--task', type=str, required=False, default="image-to-text-with-past")
argparser.add_argument('--output', type=str, required=False, default="google/pix2struct-docvqa-base")
args = argparser.parse_args()

cmd = f"optimum-cli export onnx --model {args.model} --task {args.task} {args.output}"
os.system(cmd)
