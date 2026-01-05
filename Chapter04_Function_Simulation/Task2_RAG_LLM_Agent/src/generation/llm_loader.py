"""
LLM加载器
"""
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class LLMLoader:
    """LLM加载器"""
    
    @staticmethod
    def load(config):
        """
        加载LLM模型和分词器
        
        Args:
            config: 模型配置字典
            
        Returns:
            model, tokenizer
        """
        model_name = config['llm_model_name']
        use_quantization = config.get('use_quantization', False)
        device_map = config.get('device_map', 'auto')
        torch_dtype_str = config.get('torch_dtype', 'bfloat16')
        
        # 解析torch dtype
        if torch_dtype_str == 'float32':
            torch_dtype = torch.float32
        elif torch_dtype_str == 'float16':
            torch_dtype = torch.float16
        elif torch_dtype_str == 'bfloat16':
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.bfloat16
        
        # 加载分词器
        logger.info(f"加载分词器: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 配置加载参数
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
            "low_cpu_mem_usage": True
        }
        
        # 量化配置
        if use_quantization:
            logger.info("使用4-bit量化加载模型")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["torch_dtype"] = torch_dtype
        
        # 加载模型
        logger.info(f"加载模型: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        logger.info("模型加载完成")
        logger.info(f"  设备: {next(model.parameters()).device}")
        logger.info(f"  数据类型: {next(model.parameters()).dtype}")
        
        return model, tokenizer