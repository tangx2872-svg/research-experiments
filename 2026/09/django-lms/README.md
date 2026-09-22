# Django LMS 实验

基于 [adityagoyal222/django-lms](https://github.com/adityagoyal222/django-lms) 进行本地部署、学习与二次开发的研究生实验项目。以课程、作业和学习资料管理为起点，逐步探索作业分析与自动批改。

## 功能完善（2026-09-22）

已补齐角色与课程权限、提交截止校验与更新、批改名单与评语、改分历史、课程归档及退课后记录保留。详细规则、迁移说明和验证结果见 [功能逻辑与验证记录](docs/功能逻辑与验证.md)。

本轮通过 22 项自动化测试及独立 SQLite 环境的浏览器完整流程验证。根据本机终端反馈，4 个新增 MySQL 迁移均已执行成功；MySQL 环境的完整页面流程仍待复验。日常启动见下方“已有环境：直接运行”。

## 1. 为什么选择这个项目

项目规模较小，课程、作业、用户和资源模块划分清晰，适合学习 Django 的模型、视图、模板及文件上传机制。已有教师与学生两种角色，可以围绕“教师发布作业 → 学生提交文件 → 教师评分”开展完整流程实验。

本仓库记录环境搭建、功能验证与后续修改，不将原作者的实现作为从零开发的成果。

## 2. 页面展示与使用流程

以下使用虚构的“演示教师”“演示学生”和演示课程。首页截图沿用原有演示，业务截图已更新为本次浏览器实际操作结果，使用独立 SQLite 演示数据库，未修改本机 MySQL 业务数据；不代表 MySQL 环境下的完整浏览器验收。

### 首页：进入学习平台

首页提供平台介绍和“查看全部课程”入口，用户可从课程页面继续注册或登录。

![在线学习平台首页](docs/images/01-home.png)

### 教师端：管理课程、作业与学习资料

教师创建课程后，可以查看课程说明、发布作业和上传资料。课程详情集中展示作业要求、截止时间及可下载的学习资料。

![教师端课程详情与作业、资料列表](docs/images/02-teacher-course.png)

### 学生端：提交作业

学生加入课程，打开具体作业后进入提交页面，填写提交标题、说明并选择附件。对应作业和提交人由系统自动绑定；截止前且未评分时可以更新或撤回。

![学生端作业文件提交页面](docs/images/03-student-submit.png)

### 教师端：手动评分

教师查看学生提交内容后进入评分页面，输入 0～100 的整数成绩和评语。支持修改评分并保留历史记录；自动批改属于后续探索方向。

![教师端作业评分页面](docs/images/04-teacher-grade.png)

### 学生端：查看评分结果

评分后，学生可在提交详情中查看提交说明、附件名称、评分状态和成绩。下图中的 92 分是演示流程录入的示例成绩。

![学生端查看已评分的作业](docs/images/05-student-result.png)

### 教师端：批改工作台

按班级名单查看未提交、待评分、已评分情况，直接进入提交详情或评分页面；退出课程的学生提交单独保留。

![教师端批改名单与提交状态](docs/images/06-grading-roster.png)

## 3. 当前功能

| 模块 | 当前代码中的功能 |
| --- | --- |
| 用户 | 注册、登录、退出登录、个人信息展示；区分教师和学生 |
| 课程 | 创建与编辑课程、成员名单、归档与恢复、学生加入或退出课程 |
| 作业 | 发布、修改、删除作业；设置作业要求与截止时间 |
| 提交与评分 | 提交与更新、截止校验、撤回、批改名单、成绩与评语、改分历史 |
| 学习资料 | 上传、下载和删除课程资料 |
| 界面 | 已有中文导航、页面提示与表单标签 |

当前使用课程归档代替直接删除，保留教学记录；个人信息展示不包含资料编辑。

### 关键业务规则

| 场景 | 当前规则 |
| --- | --- |
| 教师管理 | 只能管理自己课程的作业、资料和评分，后台校验归属 |
| 学生提交 | 必须已选课；提交人自动取登录账号，作业由页面地址确定 |
| 更新与撤回 | 仅截止前、未评分、课程未归档时允许；退课后不能修改 |
| 重复提交 | 每人每份作业一份有效提交；更新覆盖当前内容，保留首次提交及最近更新时间 |
| 截止时间 | 按北京时间输入和展示，截止时刻起由后台禁止提交 |
| 成绩反馈 | 0～100 整数分数及评语；修改评分追加记录，学生可查看历史 |
| 课程归档 | 保留内容和记录，停止选课及教学修改；教师可恢复 |
| 退出课程 | 保留个人提交、附件和成绩，从个人页继续查看 |
| 删除作业 | 已有学生提交时禁止删除，避免丢失教学记录 |
| 附件 | 最大 20 MB，拒绝空文件；支持常见文档、图片、ZIP、Python 和 Notebook，下载前校验权限 |

旧系统重复提交保留为历史版本，优先以已评分记录作为有效提交；旧成绩保留为初始评分记录。当前提交更新不保存每次编辑的完整快照。自动批改、退回重做、教师注册审核尚未实现，GraphQL 入口暂时关闭。

## 4. 技术栈与目录

- Python 3.8.20、Django 3.1.3（本机 `classroom` 环境已核对）
- MySQL（当前默认数据库后端；本机服务版本本次未核验）
- HTML、CSS、Bootstrap 4、Django 模板
- Graphene-Django 2.13.0（依赖声明版本；GraphQL 入口暂时关闭，等待统一权限规则）
- Windows、Conda、Git；可使用 VS Code 编辑

```text
 django-lms/
 ├── assignments/        # 作业发布、提交与评分
 ├── courses/            # 课程与选课管理
 ├── django_lms/         # Django 配置与总路由
 ├── docs/images/        # README 页面截图
 ├── resources/          # 学习资料管理
 ├── static/             # 样式与静态资源
 ├── templates/          # 公共页面模板
 ├── users/              # 用户与身份管理
 ├── manage.py           # Django 管理入口
 └── requirements.txt    # Python 依赖
```

## 5. 本地运行

### 已有环境：直接运行

本机已完成本轮数据库迁移。在之前已配置数据库密码的终端中执行即可，无需每次重新安装依赖或迁移。

Windows CMD：

```bat
conda activate classroom
cd /d D:\Research\research-experiments\2026\09\django-lms
python manage.py runserver
```

Windows PowerShell：

```powershell
conda activate classroom
Set-Location D:\Research\research-experiments\2026\09\django-lms
python manage.py runserver
```

看到 `Starting development server at http://127.0.0.1:8000/` 后，打开 [本地学习平台](http://127.0.0.1:8000/)。保持终端打开，按 `Ctrl + C` 停止服务。

如果提示找不到 `manage.py`，说明没有进入上述项目目录，也可以使用完整路径：

```powershell
python D:\Research\research-experiments\2026\09\django-lms\manage.py runserver
```

新终端如未继承 `MYSQL_PASSWORD`，需重新设置环境变量。下面是首次搭建或更新环境时的完整步骤，命令采用 PowerShell。

### 激活环境并安装依赖

```powershell
conda activate classroom
Set-Location D:\Research\research-experiments\2026\09\django-lms
python -m pip install -r requirements.txt
```

如尚未创建环境，可先执行 `conda create -n classroom python=3.8`，再激活环境。

### 准备 MySQL 数据库

确保 MySQL 服务已启动，并创建数据库，例如在 MySQL 客户端中执行：

```sql
CREATE DATABASE IF NOT EXISTS django_lms CHARACTER SET utf8mb4;
```

当前 `django_lms/settings.py` 使用 `localhost:3306`、数据库 `django_lms`、用户 `root`。如本机配置不同，请调整对应配置项。

数据库密码从 `MYSQL_PASSWORD` 环境变量读取。在启动 Django 的同一个 PowerShell 窗口中设置；以下方式避免将密码明文写进命令历史：

```powershell
$dbCredential = Get-Credential -UserName root -Message '请输入本机 MySQL 密码'
$env:MYSQL_PASSWORD = $dbCredential.GetNetworkCredential().Password
Remove-Variable dbCredential
```

当前配置不会自动加载 `.env` 文件，也不要把真实密码写进 README 或提交到仓库。

### 检查、迁移并启动

```powershell
python manage.py check
python manage.py migrate
python manage.py runserver
```

打开 [本地首页](http://127.0.0.1:8000/)。如需要 Django 管理后台账号，可执行 `python manage.py createsuperuser`；教师与学生业务账号可在注册页面创建并选择身份。

页面依赖外部 Bootstrap、图标和字体资源，首页插图也来自外部地址；网络不可用时可能影响展示。

### 手动体验业务流程

1. 注册教师账号，创建课程并发布作业，设置未来的截止时间。
2. 退出后注册学生账号，进入全部课程并加入课程。
3. 打开作业详情，填写说明并上传附件；截止前且未评分时可更新或撤回。
4. 切换回教师账号，在批改工作台查看提交名单，录入分数及评语。
5. 修改一次评分，再切换回学生账号，查看最新结果和两条评分记录。
6. 教师归档课程，确认停止修改；恢复课程后继续管理。
7. 学生退出课程，从个人页确认历史提交、附件和成绩仍可查看。

### 运行自动化验证

在项目目录中执行：

```powershell
python manage.py check
python manage.py test --settings=django_lms.test_settings
python manage.py makemigrations --check --dry-run --settings=django_lms.test_settings
```

测试使用独立 SQLite 数据库及临时附件目录，不操作 MySQL 业务数据。当前共 22 项测试，覆盖核心流程、越权、截止限制、提交更新、文件校验、评分历史、归档与恢复、退课及旧数据迁移。

## 6. 当前实验进度

- [x] 将开源项目整理至 `research-experiments/2026/09/django-lms`。
- [x] 配置 Python / Conda 环境；当前代码使用环境变量读取数据库密码。
- [x] 主要页面已有中文界面。
- [x] 本次执行 `python manage.py check`，结果为 0 个问题。
- [x] 在独立 SQLite 演示环境完成迁移，并通过测试客户端验证创建课程、发布作业、选课、文件提交、评分和成绩展示。
- [x] 整理页面截图、使用说明和项目来源。
- [x] 本机 MySQL 已成功执行 `assignments.0003`、`0004`、`0005` 及 `courses.0002`，以用户终端输出的 4 个 `OK` 为依据。
- [ ] 在本机 MySQL 环境中完成完整浏览器流程复验。
- [x] 补齐核心角色权限、截止校验、提交更新、评分历史和课程归档，并通过回归测试。
- [x] 完成一轮教学业务功能与交互完善，详细范围见功能逻辑文档。
- [ ] 探索自动批改与作业分析。

本次覆盖核心正常流程及上述权限、状态边界，不代表所有并发和部署场景均已通过验收。

## 7. 后续计划

继续完成本机 MySQL 环境复验，并按实际教学需求扩充并发场景、评分标准和提交版本策略。

后续在保留人工评分的基础上，设计自动批改的输入、评分标准和结果记录方式，并比较自动评分与人工评分结果。

## 8. 项目来源与许可记录

原项目：[adityagoyal222/django-lms](https://github.com/adityagoyal222/django-lms)。感谢原作者提供的学习管理系统实现。本目录是在其基础上的本地部署、中文界面适配和实验维护。

当前本地目录未发现 `LICENSE` 文件，本次查看的上游仓库首页也未显示许可证信息，因此暂不声明其采用 MIT License。后续需核实上游许可证或作者授权，并保留适用的版权与许可说明。
