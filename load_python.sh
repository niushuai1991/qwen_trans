# 获取脚本所在的目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# 将脚本所在目录下的py311目录添加到PATH变量最前面
export PATH=$SCRIPT_DIR/py311/bin:$PATH
