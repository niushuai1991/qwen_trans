
## 安装依赖

## 安装xformers

```
pip install xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

## 安装unsloth

```
pip3 install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```


### 安装pytorch（不需要）
当前最新版本pytorch是2.4.0,但是xformers 0.0.26.post1使用的是2.3.0，所以通过xformers安装就能安装对应版本的pytorch，不需要再单独安装。

```
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

卸载
```
pip uninstall -y torch torchvision torchaudio
```


## 检查CUDA是否可用

把以下内容保存到check_cuda.py文件里，然后执行，如果只出现3个输出证明版本不兼容，如果出现4个输出则证明兼容了。
``` python
import torch
print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(torch.cuda.current_device()))
```

## windows安装triton

https://huggingface.co/madbuda/triton-windows-builds



```
pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

## 报错修复

### Otherwise in local machines, your xformers version of 0.0.27.post2 is too new.
``` log
(venv) D:\testcode\qwen_trans>python run.py
Traceback (most recent call last):
  File "D:\testcode\qwen_trans\run.py", line 1, in <module>
    from unsloth import FastLanguageModel
  File "D:\testcode\qwen_trans\venv\lib\site-packages\unsloth\__init__.py", line 158, in <module>
    from .models import *
  File "D:\testcode\qwen_trans\venv\lib\site-packages\unsloth\models\__init__.py", line 15, in <module>
    from .loader  import FastLanguageModel
  File "D:\testcode\qwen_trans\venv\lib\site-packages\unsloth\models\loader.py", line 15, in <module>
    from ._utils import is_bfloat16_supported, HAS_FLASH_ATTENTION, HAS_FLASH_ATTENTION_SOFTCAPPING
  File "D:\testcode\qwen_trans\venv\lib\site-packages\unsloth\models\_utils.py", line 196, in <module>
    raise ImportError(
ImportError: Unsloth: If you are in Colab, we updated the top cell install instructions - please change it to below then press Disconnect Runtime and then Restart it.

%%capture
# Installs Unsloth, Xformers (Flash Attention) and all other packages!
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

Otherwise in local machines, your xformers version of 0.0.27.post2 is too new.
Please downgrade xformers via `pip install --force-reinstall "xformers<0.0.27"
```

这里我们需要把xformers降级

```
pip uninstall -y xformers
pip install xformers==0.0.26.post1 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```
安装后使用check_cuda.py脚本检查一下，如果失败
```
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


### AssertionError: Torch not compiled with CUDA enabled

``` log
(venv) D:\testcode\qwen_trans>python run.py
Traceback (most recent call last):
  File "D:\testcode\qwen_trans\run.py", line 1, in <module>
    from unsloth import FastLanguageModel
  File "D:\testcode\qwen_trans\venv\lib\site-packages\unsloth\__init__.py", line 87, in <module>
    major_version, minor_version = torch.cuda.get_device_capability()
  File "D:\testcode\qwen_trans\venv\lib\site-packages\torch\cuda\__init__.py", line 430, in get_device_capability
    prop = get_device_properties(device)
  File "D:\testcode\qwen_trans\venv\lib\site-packages\torch\cuda\__init__.py", line 444, in get_device_properties
    _lazy_init()  # will define _get_device_properties
  File "D:\testcode\qwen_trans\venv\lib\site-packages\torch\cuda\__init__.py", line 284, in _lazy_init
    raise AssertionError("Torch not compiled with CUDA enabled")
AssertionError: Torch not compiled with CUDA enabled
```



### ModuleNotFoundError: No module named '_lzma'

先在系统里安装lzma库
```
sudo apt install liblzma-dev -y
```

python中安装lzma库
```
pip install backports.lzma
```
如果还不行，是python库中的代码有问题，找到python目录中lib\python3.11\lzma.py，修改
``` python
from _lzma import *
from _lzma import _encode_filter_properties, _decode_filter_properties
```
替换为
``` python
try:
    from _lzma import *
    from _lzma import _encode_filter_properties, _decode_filter_properties
except ImportError:
    from backports.lzma import *
    from backports.lzma import _encode_filter_properties, _decode_filter_properties
```


### OSError: Incorrect path_or_model_id: './model/Qwen2-7B-Instruct'. Please provide either the path to a local folder or the repo_id of a model on the Hub.
修改lora_model\adapter_config.json 和 trans.py文件里对应的配置
