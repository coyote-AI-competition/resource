#!/bin/bash
set -e

# GithubのURL
# GITHUB_URL="https://github.com/coyote-AI-competition/resource"
EXPECTED_PYTHON_VERSION="3.12"
PYTHON_COMMAND="python"
VENV_FOLDER="venv"
ENV_MEMO="env.txt"
# カラー定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
# OS検出
detect_os() {
  case "$(uname -s)" in
    Drawin)
      echo "macOS"
      ;;
    Linux)
      echo "Linux"
      ;;
    *"_NT"*)
      echo "Windows"
      print_error "setup.batを実行してください。"
      exit 1
      ;;
    *)
      echo "Unknown"
      print_error "サポートされていないOSです"
      exit 1
  esac
}

OS_TYPE=$(detect_os)

# ヘルパー関数
print_step() {
  echo -e "${GREEN}==>${NC} $1"
}
print_info() {
  echo -e "${BLUE}INFO:${NC} $1"
}
print_warning() {
  echo -e "${YELLOW}警告:${NC} $1"
}
print_error() {
  echo -e "${RED}エラー:${NC} $1"
}
confirm() {
  read -p "$1 (y/n) " -n 1 -r
  echo
  [[ ! $REPLY =~ ^[Yy]$ ]] && return 1
  return 0
}
# バナー表示
echo -e "${GREEN}"
echo "=============================================="
echo "  Coyote AI Competition Pythonローカル環境構築  "
echo "=============================================="
echo -e "${NC}"

print_info "検出されたOS: $OS_TYPE"

# 必要な環境のチェック&構築
print_step "動作に必要な環境を確認しています..."
if ! command -v ${PYTHON_COMMAND} &> /dev/null; then
  PYTHON_COMMAND="python3"
  if ! command -v ${PYTHON_COMMAND} &> /dev/null; then
    print_error "pythonはインストールされていません。"
    exit 1
  fi
fi
print_info "pythonはインストールされています。"

print_step "Pythonのバージョンを確認しています..."
INSTALLED_VERSION=$(${PYTHON_COMMAND} --version 2>&1 | awk '{print $2}')
print_info "インストールされているバージョン: ${INSTALLED_VERSION}" 
INSTALLED_VERSION=$(echo "$INSTALLED_VERSION" | cut -c 1-4)
if [ "$INSTALLED_VERSION" = "$EXPECTED_PYTHON_VERSION" ]; then
  print_info "正しいバージョンのPythonがインストールされています。"
else
  print_error "正しいバージョンのPythonをインストールしてください。"
  exit 1
fi

print_step "Pythonの環境構築をしています..."
${PYTHON_COMMAND} -m venv ./${VENV_FOLDER}
echo "venv環境に入ります。"
source ./${VENV_FOLDER}/bin/activate

if [[ "$VIRTUAL_ENV" != "" ]]; then
  print_info "現在の仮想環境: $VIRTUAL_ENV"
else
  print_error "仮想環境はアクティブではありません。"
fi
echo "Pythonのライブラリをインストールします。"
pip install -r "./requirements.txt"
echo "環境構築が終了しました。"
echo "venv環境から退出します。"
deactivate
print_step "必要なファイルを作成しています..."
{
  echo "#!/bin/bash"
  echo ""
  echo "alias activate=\"source ./venv/bin/activate\""
  echo "alias freeze=\"pip freeze > ./requirements.txt\""
  echo "alias importreq=\"pip install -r ./requirements.txt\""
  echo "alias arena=\"${PYTHON_COMMAND} ./arena.py\""
  echo "alias client=\"${PYTHON_COMMAND} ./client/client.py\""
  echo "alias server=\"${PYTHON_COMMAND} ./server/server.py\""
  echo "activate"
} > "./setup_session.sh"
chmod 775 "./setup_session.sh"

source ./setup_session.sh

print_info "環境構築が完了しました。"
print_step "環境をメモしています..."
{
  echo "PYTHON_COMMAND=${PYTHON_COMMAND}"
  echo "INSTALL_DIR=${INSTALL_DIR}"
  echo "OS_TYPE=${OS_TYPE}"
} > "./${ENV_MEMO}"

cd "./"
echo "~~~ コマンド情報 ~~~"
echo "初回起動時"
echo "source ./setup_session.sh"
echo ""
echo "activate  : 仮想環境を有効にする"
echo "arena     : アリーナモードで起動"
echo "server    : サーバーを起動"
echo "client    : クライアントモードで起動"
echo "freeze    : requirements.txtを生成"
echo "importreq : requirements.txtからライブラリをインストール"