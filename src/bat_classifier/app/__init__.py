"""蝙蝠声学分类桌面 App。

入口：run_app()。顺序约束——必须先加载设置并写入
BAT_WORKSPACE 环境变量，再导入 PySide6 界面与库代码，
因为 config 的全部路径在导入期解析。
"""

from __future__ import annotations

import sys


def run_app(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    from .settings import AppSettings

    settings = AppSettings.load()
    settings.apply()

    # 以下导入依赖已就绪的 BAT_WORKSPACE，顺序不可提前。
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication(arguments or ["bat_classifier_app"])
    app.setApplicationName("蝙蝠声学分类系统")
    window = MainWindow(settings)
    window.show()

    return app.exec()
