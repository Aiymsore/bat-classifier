# 数据目录

原训练数据
链接：https://pan.quark.cn/s/8d7aed98acbf?pwd=3FPD
提取码：3FPD

本地运行时，将数据按以下任一方式放置：

1. 把 `声波文件.zip` 放到 `data/`；程序会自动解压到 `data/raw/`。
2. 直接把 `.wav` 文件放到 `data/raw/`。

建议只在 `data/sample/` 中提交少量、已脱敏且确认可公开的示例音频。完整数据可放在学校网盘、百度网盘、Google Drive 或公开数据集平台，并在主 README 中说明获取方式与授权。

## 仓库内置样例

`sample/demo_xianren_0_4s.wav` 是从现有录音中截取的 4 秒片段，只用于运行 `python demo.py --self-test`。文件采样率为 250 kHz，单声道，PCM 16-bit，体积约 2 MB。

提交公开仓库前，仍应确认原始录音的数据授权与公开范围。
