import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

import onnx
import onnxruntime
from onnx import numpy_helper
from optimum.onnxruntime import AutoOptimizationConfig, ORTOptimizer, ORTModelForSequenceClassification

import os
import numpy as np
from loguru import logger


class OnnxConverter:
    def __init__(self, ft_model: str, save_dir: str):
        self.model_name = ft_model
        self.onnx_models_folder = save_dir
        os.makedirs(self.onnx_models_folder, exist_ok=True)

        self.config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        
    def to_onnx(self, sample_text: str) -> str:
        encoding = self.tokenizer(sample_text, return_tensors='pt', padding=True, max_length=256, truncation=True)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        onnx_model_name = self.model_name.split('/')[-1]

        input_names = ["input_ids", "attention_mask"]
        dynamic_axes = {
            "input_ids" : {0: "batch_size", 1: "sequence_length"},
            "attention_mask" : {0: "batch_size", 1: "sequence_length"},
            "logits" : {0: 'batch'},
        }
        inputList = [input_ids, attention_mask]
        self.model.eval()
        torch.onnx.export(
            self.model,
            args=tuple(inputList),
            f=f"{self.onnx_models_folder}/{onnx_model_name}.onnx",
            verbose=False,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
        )
        logger.info("Export to ONNX successfully")
        return onnx_model_name


    def fix_onnx_fp16(self, onnx_model: str) -> str:
        if not os.path.exists(onnx_model):
            raise FileNotFoundError(f"ONNX model file not found: {onnx_model}")
        
        onnx_model_name = onnx_model.split('/')[-1]

        finfo = np.finfo(np.float16)
        fp16_max = finfo.max
        fp16_min = finfo.min
        model = onnx.load(onnx_model)

        fp16_fix = False
        for tensor in model.graph.initializer:
            nptensor = numpy_helper.to_array(tensor)
            if nptensor.dtype == np.float32 and (
                np.any(nptensor > fp16_max) or np.any(nptensor < fp16_min)
            ):
                logger.info(f"Tensor {tensor.name}: {nptensor} out of FP16 range")
                nptensor = np.clip(nptensor, fp16_min, fp16_max).astype(np.float16)
                new_tensor = numpy_helper.from_array(nptensor, tensor.name)
                tensor.CopyFrom(new_tensor)
                fp16_fix = True
                
        if fp16_fix:
            logger.info("Found constants out of FP16 range, clipped to FP16 range")
            onnx_model_name += "_fix_outofrange_fp16"
            onnx.save(model, f=f"{self.onnx_models_folder}/{onnx_model_name}.onnx")
            logger.info(f"Saving modified onnx file at {self.onnx_models_folder}/{onnx_model_name}.onnx")
        else:
            logger.info("Model already in FP16")
        
        try:
            session = onnxruntime.InferenceSession(
                f"{self.onnx_models_folder}/{onnx_model_name}",
                providers=["CPUExecutionProvider"]
            )
            logger.info("=== Model I/O Details ===")
            logger.info("Inputs:")
            for x in session.get_inputs():
                print(f"  {x}")
            
            logger.info("Outputs:")
            for x in session.get_outputs():
                print(f"  {x}")

        except Exception as e:
            logger.error(f"Failed to load ONNX model for I/O details: {e}")

    @staticmethod
    def convert_to_onnx(model_name: str, save_dir: str) -> str:
        logger.info(f"Exporting model {model_name} to ONNX")
        model = ORTModelForSequenceClassification.from_pretrained(model_name, export=True)
        
        optimizer = ORTOptimizer.from_pretrained(model)
        optimization_config = AutoOptimizationConfig.O4(use_raw_attention_mask=True)
        optimizer.optimize(save_dir=save_dir, optimization_config=optimization_config)
        
        model.save_pretrained(save_dir)
        logger.info(f"Exported ONNX model to {save_dir}")
        
        return save_dir

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--fine_tuned_model", type=str, default="models/classification-phobert-base-v2", required=True)
#     parser.add_argument("--output_dir", type=str, default="./onnx_models", help="Output directory to save ONNX model")
#     parser.add_argument("--sample_text", type=str, default="alo ạ vâng cháu giao đơn hàng này chú ơi chú ra cổng nhận cháu đơn hàng đây này", help="Sample text to convert ONNX model")
#     args = parser.parse_args()

#     convert = OnnxConverter(
#         ft_model=args.fine_tuned_model,
#         save_dir=args.output_dir
#     )
    
#     onnx_model = convert.to_onnx(args.sample_text)
#     convert.fix_onnx_fp16(f'{args.output_dir}/{onnx_model}.onnx')
