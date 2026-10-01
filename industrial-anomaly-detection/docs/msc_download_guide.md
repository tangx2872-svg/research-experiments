# MSC-AD 下载操作指引（nbsdc 镜像）

> 目标：从 nbsdc 国内镜像拿到 MSC-AD 数据，交给 `prepare_msc_inventory.py` 生成 dataset inventory。
> 全程**不训练模型、不改 PatchCore**。

## 为什么需要你操作

MSC-AD 是封闭分发数据集：
- 官方页（msc-ad.github.io）无公开下载链接，需大学邮箱发邮件给 `mscad2023@gmail.com` 申请密码；
- nbsdc 国内镜像（国家基础学科公共科学数据中心）需**注册账号 + 登录 + 申请下载**。

这两步都涉及你的账号/邮箱/个人信息授权，AI 无法代操作。

## nbsdc 镜像下载步骤

1. 打开数据集详情页：
   `https://nbsdc.cn/general/dataDetail?id=666070de195d266d328f4836&type=1`
   - 名称：面向工业场景的异常检测模型数据（MSC-AD）
   - 机构：复旦大学（张文强团队 = 论文作者）
   - CSTR:16666.11.nbsdc.zT8h9kUc
   - 数据量 1.07GB / 10888 文件

2. 注册账号并登录（页面右上角，需要手机号/邮箱等实名信息，这是国内科研数据平台的常规要求）。

3. 找到数据文件的**下载入口**，按平台流程申请/获取下载权限。

4. 下载数据包到本地，例如放到：
   `D:/Research/research-experiments/industrial-anomaly-detection/data/msc_ad/`

## 拿到数据后，交给我（任选其一）

在对话里告诉我下面任意一种情况，我立即跑 inventory：

- **已下载到本地压缩包**：告诉我压缩包完整路径，我执行
  ```bash
  python scripts/prepare_msc_inventory.py --archive "<压缩包路径>"
  ```
- **已解压成目录**：告诉我目录路径，我执行
  ```bash
  python scripts/prepare_msc_inventory.py --dir "<目录路径>"
  ```
- **拿到了可直链下载 URL**：告诉我 URL，我执行
  ```bash
  python scripts/prepare_msc_inventory.py --url "<下载URL>"
  ```

## inventory 会输出什么

在 `data/msc_ad/` 下生成：
- `inventory.json` — 结构化统计（surface / defect / illumination / resolution / 图片数 / mask 数）
- `inventory_report.md` — 可读报告 + 待人工确认项清单

之后据此回答：6 种表面名、5 种缺陷类型名、光照/分辨率具体值、train/test 划分、能否接 Anomalib，以及推荐子集。

## 注意

- 数据仅限研究用途，不传播、不商用（数据集许可协议要求）。
- 若 nbsdc 下载受阻，备选：用大学邮箱走官方 `mscad2023@gmail.com` 申请。
