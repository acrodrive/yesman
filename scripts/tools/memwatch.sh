#!/usr/bin/env bash
# 명령 하나를 실행하면서 컨테이너(cgroup) 전체의 메모리를 1초마다 재고, 끝나면 걸린 시간과 최대 메모리를 출력한다.
# ray worker는 별도 프로세스라 프로세스 하나의 RSS로는 잴 수 없으므로 cgroup의 anon 메모리(페이지 캐시 제외)를 본다.
# 사용법: scripts/tools/memwatch.sh <명령> [인자...]
set -u
peak=0
(
  while true; do
    a=$(awk '$1=="anon"{print $2}' /sys/fs/cgroup/memory.stat)
    echo "$a"
    sleep 1
  done
) > /tmp/memwatch.$$ &
watcher=$!
start=$(date +%s)
"$@"
rc=$?
end=$(date +%s)
kill $watcher 2>/dev/null
peak=$(sort -n /tmp/memwatch.$$ | tail -1)
base=$(head -1 /tmp/memwatch.$$)
rm -f /tmp/memwatch.$$
echo "[memwatch] exit=$rc elapsed_s=$((end-start)) peak_anon_GB=$(awk -v p=$peak 'BEGIN{printf "%.2f", p/1e9}') start_anon_GB=$(awk -v p=$base 'BEGIN{printf "%.2f", p/1e9}')"
exit $rc
