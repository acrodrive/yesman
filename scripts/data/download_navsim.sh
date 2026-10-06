#!/usr/bin/env bash
# NAVSIM 데이터 다운로드 (yesman.md 11절 2단계 스크립트를 이 저장소의 경로 규칙에 맞춘 것).
# 결과는 NAVSIM 공식 구조로 바로 저장한다:
#   /workspace/datasets/navsim/{maps, navsim_logs/{trainval,test}, sensor_blobs/{trainval,test}/<log>/CAM_{L0,F0,R0}}
#
# 사용법 (source scripts/setup/env.sh 후):
#   download_navsim.sh maps logs_test navtest_keep "testcam 0"            # 1단계 확인용
#   download_navsim.sh maps logs_test logs_trainval navtest_keep \
#       "testcam $(seq -s' ' 0 31)" "navtrain 1 5 9 13 17 21 25 29"     # 2단계 전체
# 각 조각은 끝나면 _done/ 에 표시하므로, 중간에 끊겨도 같은 명령으로 이어서 받는다.
set -uo pipefail

ROOT=${OPENSCENE_DATA_ROOT:-/workspace/datasets/navsim}
DEVKIT=${NAVSIM_DEVKIT_ROOT:-/workspace/yesman/third_party/navsim}
# 압축을 푸는 임시 폴더는 컨테이너 디스크에 둔다. 남길 파일만 볼륨으로 옮기므로,
# pod가 꺼져도 잃는 것은 진행 중이던 한 조각뿐이다(_done 표시가 없으므로 다시 받는다).
TMP=${NAVSIM_DL_TMP:-/tmp/navsim_dl}
HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main
CAMS=('*/CAM_F0/*' '*/CAM_L0/*' '*/CAM_R0/*')
mkdir -p "$ROOT"/{_done,navsim_logs,sensor_blobs/test,sensor_blobs/trainval} "$TMP" && cd "$ROOT"

# 스트리밍으로 받으면서 전방 3캠만 푼다. 실패하면 그 조각만 다시 받는다.
get_cams () {  # $1=URL $2=이름
  for t in 1 2 3; do
    rm -rf "$TMP"/*
    if curl -fsSL "$1" | tar -xz -C "$TMP" --wildcards "${CAMS[@]}"; then return 0; fi
    echo "retry $2 ($t)"
  done; echo "FAILED $2"; return 1
}

for job in "$@"; do
  set -- $job
  case $1 in
  maps)  # nuPlan 지도. S3가 연결당 약 40KB/s로 느려서, 64개 구간으로 나눠 병렬로 받는다
    [ -d maps ] || {
      URL=https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
      SIZE=$(curl -sI $URL | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
      N=64; CH=$(( (SIZE + N - 1) / N )); mkdir -p "$TMP/maps_parts"
      for k in $(seq 0 $((N-1))); do
        s=$((k*CH)); e=$(( s+CH-1 < SIZE-1 ? s+CH-1 : SIZE-1 ))
        curl -fsS --retry 5 -r $s-$e -o "$TMP/maps_parts/$(printf %03d $k)" $URL &
      done; wait
      cat "$TMP"/maps_parts/* > "$TMP/maps.zip" && rm -rf "$TMP/maps_parts"
      [ "$(stat -c %s "$TMP/maps.zip")" = "$SIZE" ] && unzip -tq "$TMP/maps.zip" \
        && unzip -q "$TMP/maps.zip" -d "$TMP" && mv "$TMP/nuplan-maps-v1.0" maps && rm -f "$TMP/maps.zip" \
        || echo "FAILED maps"
    } ;;
  logs_test|logs_trainval)  # 로그 (4~6단계는 지도와 로그만 있으면 된다)
    s=${1#logs_}
    [ -d navsim_logs/$s ] || { curl -fsSL $HF/openscene-v1.1/openscene_metadata_${s}.tgz | tar -xz -C "$TMP" \
      && mv "$TMP/openscene-v1.1/meta_datas/$s" navsim_logs/$s && rm -rf "$TMP/openscene-v1.1"; } ;;  # 아카이브 안은 meta_datas/<split>/*.pkl
  navtest_keep)  # navtest 장면의 현재 프레임 3캠 이미지 목록 (sensor_blobs/test 기준 상대 경로)
    [ -f navtest_keep.txt ] || DEVKIT=$DEVKIT python - <<'EOF'
import glob, os, pickle, yaml
cfg = os.environ["DEVKIT"] + "/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
tokens = set(yaml.safe_load(open(cfg))["tokens"])
keep, found = set(), set()
for f in glob.glob("navsim_logs/test/**/*.pkl", recursive=True):
    for frame in pickle.load(open(f, "rb")):
        if frame["token"] in tokens:
            found.add(frame["token"])
            for cam in ("CAM_L0", "CAM_F0", "CAM_R0"):
                keep.add(frame["cams"][cam]["data_path"])
open("navtest_keep.txt", "w").write("\n".join(sorted(keep)) + "\n")
print(len(tokens), "navtest tokens,", len(found), "found in logs,", len(keep), "images")
EOF
    ;;
  testcam)  # navtest: test 카메라 아카이브만 받고(LiDAR 제외), 목록에 있는 이미지만 남긴다
    shift
    for i in "$@"; do
      [ -e _done/test_cam_$i ] && continue
      get_cams $HF/openscene-v1.1/openscene_sensor_test_camera/openscene_sensor_test_camera_${i}.tgz test_cam_$i \
        && rsync -a --ignore-missing-args --files-from=navtest_keep.txt \
             "$TMP/openscene-v1.1/sensor_blobs/test/" sensor_blobs/test/ \
        && touch _done/test_cam_$i && echo "done test_cam_$i"
    done ;;
  navtrain)  # navtrain: current 조각만 받는다(history 불필요). 조각 번호는 1~32
    shift
    for i in "$@"; do
      [ -e _done/navtrain_cur_$i ] && continue
      get_cams $HF/navsim/navtrain_current_${i}.tgz navtrain_cur_$i \
        && rsync -a "$TMP/navtrain_current_${i}/" sensor_blobs/trainval/ \
        && ls "$TMP/navtrain_current_${i}" > _done/navtrain_cur_$i && echo "done navtrain_cur_$i"
    done ;;
  *) echo "unknown job: $1"; exit 1 ;;
  esac
done
rm -rf "$TMP"
