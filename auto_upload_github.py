import os
import subprocess
from pathlib import Path


# =========================
# 1. 这里写死你的 Git 身份信息
# =========================
# 以后如果你要换 GitHub 账号，改这里就行
GIT_USER_NAME = "anancat63"
GIT_USER_EMAIL = "2914305646qhd@gmail.com"
github_username = "anancat63"

# =========================
# 2. 默认 .gitignore 内容
# =========================
DEFAULT_GITIGNORE = """# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd

# Virtual env
.venv/
venv/
env/

# IDE
.vscode/
.idea/

# Logs
logs/
*.log

# Build output
build/
dist/
*.egg-info/

# Environment
.env

# OS files
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Large files
*.pt
*.pth
*.bin
*.ckpt
*.safetensors

# Node
node_modules/
"""


def run_cmd(cmd, cwd=None, check=True):
    """
    执行命令，并打印输出

    参数说明：
    - cmd: 命令列表，例如 ["git", "status"]
    - cwd: 命令执行目录
    - check: 如果命令失败，是否抛异常

    返回值：
    - subprocess.CompletedProcess
    """
    print(f"\n[执行命令] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        shell=False
    )

    # 打印标准输出
    if result.stdout:
        print(result.stdout.strip())

    # 打印错误输出
    if result.stderr:
        print(result.stderr.strip())

    # 根据需要决定是否报错
    if check and result.returncode != 0:
        raise RuntimeError(f"命令执行失败：{' '.join(cmd)}")

    return result


def check_git_installed():
    """
    检查本机是否已经安装 Git
    """
    try:
        run_cmd(["git", "--version"], check=True)
    except Exception:
        raise EnvironmentError("未检测到 Git，请先安装 Git，并确保 git 已加入系统 PATH 环境变量。")


def ensure_git_repo(project_dir: Path):
    """
    检查当前目录是否已经是 Git 仓库：
    - 如果存在 .git，说明已经初始化过
    - 如果不存在 .git，则自动执行 git init
    """
    git_dir = project_dir / ".git"

    if git_dir.exists() and git_dir.is_dir():
        print("\n[信息] 当前目录已经是本地 Git 仓库（检测到 .git）。")
    else:
        print("\n[信息] 当前目录不是 Git 仓库，正在执行 git init ...")
        run_cmd(["git", "init"], cwd=project_dir)


def ensure_gitignore(project_dir: Path):
    """
    检查 .gitignore 是否存在：
    - 存在：直接使用
    - 不存在：自动创建默认版
    """
    gitignore_path = project_dir / ".gitignore"

    if gitignore_path.exists():
        print("\n[信息] 已检测到 .gitignore，无需创建。")
    else:
        print("\n[信息] 未检测到 .gitignore，正在自动创建默认 .gitignore ...")
        gitignore_path.write_text(DEFAULT_GITIGNORE, encoding="utf-8")
        print(f"[信息] 已创建：{gitignore_path}")


def set_git_user_config():
    """
    自动设置 Git 全局用户名和邮箱
    这里使用脚本顶部写死的 GIT_USER_NAME / GIT_USER_EMAIL
    """
    run_cmd(["git", "config", "--global", "user.name", GIT_USER_NAME])
    run_cmd(["git", "config", "--global", "user.email", GIT_USER_EMAIL])

    # 再打印出来确认
    run_cmd(["git", "config", "--global", "user.name"])
    run_cmd(["git", "config", "--global", "user.email"])


def has_commit_history(project_dir: Path):
    """
    判断当前仓库是否已经有提交历史

    方法：
    - 执行 git rev-parse --verify HEAD
    - 成功：说明已经有至少一个 commit
    - 失败：说明还没有 commit
    """
    result = run_cmd(["git", "rev-parse", "--verify", "HEAD"], cwd=project_dir, check=False)
    return result.returncode == 0


def has_uncommitted_changes(project_dir: Path):
    """
    判断当前目录是否有未提交的改动

    方法：
    - git status --porcelain
    - 有输出：说明有变更
    - 无输出：说明没有变更
    """
    result = run_cmd(["git", "status", "--porcelain"], cwd=project_dir, check=False)
    return bool(result.stdout.strip())


def commit_changes(project_dir: Path):
    """
    自动 add + commit

    提交策略：
    - 如果没有提交历史，commit message 用 "initial commit"
    - 如果已有提交历史，commit message 用 "update project"
    - 如果没有任何改动，则跳过 commit
    """
    run_cmd(["git", "add", "."], cwd=project_dir)

    # 检查是否有改动
    if not has_uncommitted_changes(project_dir):
        print("\n[信息] 没有检测到新改动，无需提交。")
        return

    # 判断是首次提交还是后续更新
    if has_commit_history(project_dir):
        commit_message = "update project"
    else:
        commit_message = "initial commit"

    print(f"\n[信息] 正在提交，提交信息：{commit_message}")
    run_cmd(["git", "commit", "-m", commit_message], cwd=project_dir)


def ensure_main_branch(project_dir: Path):
    """
    将当前分支统一设为 main

    说明：
    - 现代 GitHub 仓库默认主分支一般是 main
    - 这一步可以避免 master/main 混乱
    """
    print("\n[信息] 正在统一主分支名称为 main ...")
    run_cmd(["git", "branch", "-M", "main"], cwd=project_dir)


def ensure_remote(project_dir: Path, github_username: str, repo_name: str):
    """
    确保远程 origin 已配置

    逻辑：
    - 如果没有 origin：自动添加
    - 如果已有 origin：自动更新为新的地址

    最终远程地址格式：
    https://github.com/用户名/仓库名.git
    """
    remote_url = f"https://github.com/{github_username}/{repo_name}.git"

    # 查看当前所有远程仓库
    result = run_cmd(["git", "remote"], cwd=project_dir, check=False)
    remotes = result.stdout.split() if result.stdout else []

    if "origin" in remotes:
        print("\n[信息] 已检测到远程 origin，正在更新地址 ...")
        run_cmd(["git", "remote", "set-url", "origin", remote_url], cwd=project_dir)
    else:
        print("\n[信息] 未检测到远程 origin，正在添加 origin ...")
        run_cmd(["git", "remote", "add", "origin", remote_url], cwd=project_dir)

    print(f"[信息] 当前远程仓库地址：{remote_url}")


def push_to_github(project_dir: Path):
    """
    将代码 push 到 GitHub

    说明：
    - 首次 push 可能需要浏览器授权登录
    - 如果远程仓库不存在，这一步会失败
    - 所以你必须提前在 GitHub 上创建一个空仓库
    """
    print("\n[信息] 正在推送到 GitHub ...")
    run_cmd(["git", "push", "-u", "origin", "main"], cwd=project_dir)


def main():
    """
    主函数流程：

    1. 获取当前脚本所在目录，作为项目目录
    2. 输入 GitHub 用户名和仓库名
    3. 检查 Git 是否安装
    4. 检查/初始化 .git
    5. 检查/创建 .gitignore
    6. 设置 Git 用户名邮箱
    7. add + commit
    8. 统一 main 分支
    9. 配置远程仓库
    10. push 到 GitHub
    """

    print("=" * 60)
    print("GitHub 自动上传/更新脚本")
    print("=" * 60)

    # 当前脚本所在目录，就是项目目录
    project_dir = Path(__file__).resolve().parent
    print(f"\n[信息] 当前项目目录：{project_dir}")

    print("\n请先确保：")
    print("1. 你已经在 GitHub 上手动创建了一个空仓库")
    print("2. 当前目录就是你要上传的项目目录")
    print("3. 如果首次 push，Git 可能会弹浏览器让你登录授权\n")

    # github_username = input("请输入 GitHub 用户名：").strip()
    repo_name = input("请输入 GitHub 仓库名：").strip()

    # if not github_username:
    #     raise ValueError("GitHub 用户名不能为空。")

    if not repo_name:
        raise ValueError("GitHub 仓库名不能为空。")

    # 执行主流程
    check_git_installed()
    ensure_git_repo(project_dir)
    ensure_gitignore(project_dir)
    set_git_user_config()
    commit_changes(project_dir)
    ensure_main_branch(project_dir)
    ensure_remote(project_dir, github_username, repo_name)
    push_to_github(project_dir)

    print("\n" + "=" * 60)
    print("操作完成。")
    print("如果 GitHub 仓库页面还没显示，请刷新页面查看。")
    print("=" * 60)


if __name__ == "__main__":
    main()