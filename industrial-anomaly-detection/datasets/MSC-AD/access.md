# MSC-AD 数据访问追踪记录（access.md）

> 用途：记录 MSC-AD 数据集的申请与下载全过程，论文写「数据来源 / Data Availability」章节时直接引用，避免遗忘。

## 数据集基本信息

| 字段 | 值 |
|---|---|
| 数据集名称 | MSC-AD（Multi-Scene Unsupervised Anomaly Detection Dataset for Small Defect Detection of Casting Surface） |
| 简称 | MSC-AD |
| 数据内容 | 大型铸件表面缺陷，6 种表面 × 5 类缺陷，12 种成像场景（3 档光照 × 4 档分辨率） |
| 标注 | sample-level + pixel-level 精确 ground truth |
| 数据量 | 1.07 GB，10888 个文件 |
| 数据格式 | bmp / png |
| 引用论文 | Zhao, Q., Wang, Y., Wang, B., Lin, J., Yan, S., Song, W., Liotta, A., Yu, J., Gao, S., Zhang, W. "MSC-AD: A Multiscene Unsupervised Anomaly Detection Dataset for Small Defect Detection of Casting Surface." *IEEE Transactions on Industrial Informatics*, 20(4), pp. 6041–6052, 2024. DOI: 10.1109/TII.2023.3341259 |
| 官方主页 | https://msc-ad.github.io/MSC-AD2023 |
| 创建机构 | 复旦大学（张文强团队，即论文作者团队） |

## 数据获取渠道

| 渠道 | 方式 | 状态 |
|---|---|---|
| 官方主页 | 大学邮箱发邮件至 `mscad2023@gmail.com` 申请密码 | 备选 |
| nbsdc 国内镜像 | 国家基础学科公共科学数据中心，注册+申请下载 | **主选** |
| nbsdc 详情页 | https://nbsdc.cn/general/dataDetail?id=666070de195d266d328f4836&type=1 | — |
| nbsdc 科技资源标识 | CSTR:16666.11.nbsdc.zT8h9kUc | — |

## 申请与下载进度

| 日期 | 操作 | 状态 | 备注 |
|---|---|---|---|
| 2026-10-01 | 数据侦察完成，确认 MSC-AD 为 Experiment 2 候选数据 | ✅ 完成 | 产出 `docs/msc_dataset_analysis.md` 等 3 份侦察文档 |
| 2026-10-01 | 准备下载流水线脚本 `scripts/prepare_msc_inventory.py` 并自测 | ✅ 完成 | 支持 --url/--archive/--dir 三种入口 |
| — | nbsdc 注册账号 | ⏳ 待操作 | 需用户本人完成 |
| — | nbsdc 申请下载 | ⏳ 待操作 | 需用户本人完成 |
| — | 下载数据包到 `data/msc_ad/` | ⏳ 待下载 | 1.07GB |
| — | 运行 inventory 生成统计 | ⏳ 待数据 | `python scripts/prepare_msc_inventory.py --archive "<路径>"` |

## 官方邮箱申请模板（备选渠道，若 nbsdc 受阻）

邮件主题：`Application to Download the MSC-AD Dataset`

正文需包含：
1. Name（姓名）
2. Affiliation（所属大学）
3. Department（院系）
4. Position（职位）
5. Email（大学邮箱）
6. Supervisor（导师，学生必填）
7. Supervisor Email（导师邮箱，如填 6 必填）

并附同意声明：
> I have read and agree to the terms and conditions specified in the MSC-AD database webpage.
> This database will only be used for research purposes.
> I will not make any part of this database available to a third party.
> I'll not sell any part of this database or make any profit from its use.

## 论文引用（Data Availability 段落可直接套用）

> The MSC-AD dataset [Zhao et al., 2024] was used under a research-only license, obtained from the dataset's official distribution channel (msc-ad.github.io) / the National Basic Science Data Center (nbsdc.cn, CSTR:16666.11.nbsdc.zT8h9kUc). The dataset is not redistributed.

## 使用许可约束

- 仅限研究用途，不得再分发、不得出售、不得商用。
- 数据仅存储在本项目 `data/msc_ad/`，不对外传播。
