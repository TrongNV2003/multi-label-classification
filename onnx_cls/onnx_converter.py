import argparse

from onnx_cls.onnx_provider import OnnxProvider

parser = argparse.ArgumentParser()
parser.add_argument("--fine_tuned_model", type=str, default="models/classification-phobert-base-v2")
parser.add_argument("--output_dir", type=str, default="./onnx_models")
parser.add_argument("--sample_text", type=str, default="alo ạ vâng cháu giao đơn hàng này chú ơi chú ra cổng nhận cháu đơn hàng đây này")
args = parser.parse_args()

convert = OnnxProvider(args.fine_tuned_model, args.output_dir, args.sample_text)

onnx_model = convert.to_onnx()
fp16_model = convert.fix_onnx_fp16(f'{args.output_dir}/{onnx_model}.onnx')
