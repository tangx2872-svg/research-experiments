# 数据搬运命令备忘（服务器 → 本机）

> 日期 2026-10-23 ｜ 用途：把 DINOv2 线的输入数据从 AutoDL 服务器搬到 Windows 本机。
> 工具：Rclone v1.75.1（SFTP 后端），支持断点续传；断线后重跑同一命令即可自动跳过已传文件。

## 连接信息（勿硬编码密码）

- 主机：`connect.nmb2.seetacloud.com`
- 端口：`46523`
- 用户：`root`
- 远程根：`/root/autodl-tmp/research-experiments/industrial-anomaly-detection`

## 一次性：建立远程配置（密码加密存储，不进命令行历史）

```bash
rclone config create autodl sftp \
  host connect.nmb2.seetacloud.com \
  port 46523 \
  user root \
  pass "$(rclone obscure '<密码>')"
```

## 传输命令（`autodl:` 前缀即上面的配置）

```bash
P=$(command -v rclone || echo "C:/Users/Lenovo/AppData/Local/Microsoft/WinGet/Packages/Rclone.Rclone_Microsoft.Winget.Source_8wekyb3d8bbwe/rclone-v1.75.1-windows-amd64/rclone.exe")
SRC="autodl:/root/autodl-tmp/research-experiments/industrial-anomaly-detection"
DST="D:/Research/research-experiments/industrial-anomaly-detection"

# 1) M2AD 输入缓存（5.35G / 27690 文件）
"$P" copy -P --transfers 8 --checkers 16 "$SRC/data/m2ad/e7_cache/l0_448" "$DST/data/m2ad/e7_cache/l0_448"

# 2) RobustAD 数据（6.39G / 2693 文件）
"$P" copy -P --transfers 8 --checkers 16 "$SRC/data/robustad" "$DST/data/robustad"

# 3) DINOv2 权重（418M，放仓库外）
"$P" copy -P --transfers 8 --checkers 16 "autodl:/root/.cache/torch" "D:/Research/.cache/torch"

# 4~6) 校验清单（很小，放 /tmp）
"$P" copy -P "autodl:/tmp/manifest_l0_448.txt"  /tmp/
"$P" copy -P "autodl:/tmp/manifest_robustad.txt" /tmp/
"$P" copy -P "autodl:/tmp/manifest_torch.txt"    /tmp/
```

## 传完校验（对账，三份都应 missing=0 extra=0 mismatch=0）

```bash
python scripts/paper_supplement_20261023/sha256_manifest.py verify \
  "$DST/data/m2ad/e7_cache/l0_448" /tmp/manifest_l0_448.txt

python scripts/paper_supplement_20261023/sha256_manifest.py verify \
  "$DST/data/robustad" /tmp/manifest_robustad.txt

python scripts/paper_supplement_20261023/sha256_manifest.py verify \
  "D:/Research/.cache/torch" /tmp/manifest_torch.txt
```

## 注意事项

1. 本机 `rclone` 命令名可能指向坏的 winget Links 软链接，报「不是有效应用程序」时改用上面的 `$P` 完整路径。
2. 数据/权重在 `.gitignore` 排除范围内，**不要** `git add` 这些目录。
3. 清单 `/tmp/manifest_*.txt` 是传输工件，不入 git。
