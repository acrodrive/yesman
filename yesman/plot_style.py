"""그림 공통 설정: 한글 글꼴(나눔고딕, /workspace/.cache/fonts)을 matplotlib에 등록한다.

글꼴 파일은 네트워크 볼륨에 둔다(pod를 다시 띄워도 남는다). 없으면 받는 방법:
  curl -fsSL -o /workspace/.cache/fonts/NanumGothic-Regular.ttf \
    https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf
"""

import glob
import os

from matplotlib import font_manager, rcParams

FONT_DIR = os.path.join(os.environ.get("WORKSPACE", "/workspace"), ".cache/fonts")


def use_korean_font():
    for f in glob.glob(os.path.join(FONT_DIR, "NanumGothic-*.ttf")):
        font_manager.fontManager.addfont(f)
    rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
