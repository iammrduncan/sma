#!/usr/bin/env bash
# Isolate the KernelBench evaluation service on the harness GB10 (README.md, "On the harness GB10").
#
# The GB10 runs SMA's harness and KernelBench side by side. This script gives KernelBench its own
# CPUs, memory, GPU and interrupts, so the harness cannot disturb its timings:
#   - CPUs. KB_CPUS (cluster B: Cortex-A725 10-14 for compiles, Cortex-X925 15-19 for timing) belong
#     to kernelbench.slice alone. system.slice, user.slice, init.scope and sma-harness.slice run on
#     HARNESS_CPUS.
#   - Memory. On a GB10 the GPU's memory is the machine's RAM, and GPU allocations are charged to no
#     cgroup, so no limit on KernelBench can protect it. Everything else is capped instead
#     (sma-harness.slice, system.slice, user.slice), which leaves KB_MEMORY_MIN for KernelBench.
#   - GPU. kernelbench-gpu-clocks.service locks the clock and sets compute mode EXCLUSIVE_PROCESS.
#   - Interrupts. Device IRQs move to HARNESS_CPUS now and at every boot
#     (kernelbench-irq-affinity.service). irqbalance, if installed, is banned from KB_CPUS.
# It also installs SMA's dispatcher unit, sma-dispatcher.service, without starting it.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sudo bash host-setup.sh            Apply the isolation (idempotent), then check it.
  bash host-setup.sh --check         Report the isolation as it is. Exit 0 if all is ok, 1 if not.
  bash host-setup.sh --check --json  The same as a JSON list of {name, ok, detail}.
  bash host-setup.sh --print         Print every file apply would write, with its path.
  host-setup.sh --irqs               Move IRQs off KB_CPUS (root). kernelbench-irq-affinity.service
                                     runs this at boot from the installed copy.

Parameters are environment variables (pass them through sudo: sudo KB_MEMORY_MIN=80G bash ...):
  KB_CPUS=10-19            KernelBench's CPUs (kernelbench.slice)
  KB_TIMING_CPUS=15-19     where KernelBench times kernels (inside KB_CPUS)
  KB_COMPILE_CPUS=10-14    where KernelBench compiles them (inside KB_CPUS)
  HARNESS_CPUS=0-9         everything else
  KB_MEMORY_MIN=80G        memory kept for KernelBench (kernelbench.slice MemoryMin)
  HARNESS_MEMORY_MAX=32G   sma-harness.slice MemoryMax
  HARNESS_MEMORY_HIGH=30G  sma-harness.slice MemoryHigh (reclaim and throttle above this)
  SYSTEM_MEMORY_MAX=3G     system.slice MemoryMax (system services, dockerd)
  USER_MEMORY_MAX=3G       user.slice MemoryMax (login sessions)
  GPU=0  CLOCK_MHZ=2200    the GPU (nvidia-smi -i) and its locked SM clock
  SMA_USER=$SUDO_USER      the user the dispatcher runs as (never root)
  SMA_REPO=<this repo>     the SMA checkout the dispatcher runs from
  SMA_DISPATCHER_CONFIG=.sma/v100/sma-v100.yaml   its config, relative to SMA_REPO
  INSTALL_DISPATCHER=1     0 skips sma-dispatcher.service
  ALLOW_DISPATCHER_CONTROL=0   1 lets SMA_USER start, stop, and restart sma-dispatcher.service with sudo
                           and no password (/etc/sudoers.d/sma-dispatcher), and nothing else. The
                           dispatcher has to run in sma-harness.slice, which needs root to start.
EOF
}

SELF=$(readlink -f "${BASH_SOURCE[0]}")
HERE=$(dirname "$SELF")

KB_CPUS="${KB_CPUS:-10-19}"
KB_TIMING_CPUS="${KB_TIMING_CPUS:-15-19}"
KB_COMPILE_CPUS="${KB_COMPILE_CPUS:-10-14}"
HARNESS_CPUS="${HARNESS_CPUS:-0-9}"
KB_MEMORY_MIN="${KB_MEMORY_MIN:-80G}"
HARNESS_MEMORY_MAX="${HARNESS_MEMORY_MAX:-32G}"
HARNESS_MEMORY_HIGH="${HARNESS_MEMORY_HIGH:-30G}"
SYSTEM_MEMORY_MAX="${SYSTEM_MEMORY_MAX:-3G}"
USER_MEMORY_MAX="${USER_MEMORY_MAX:-3G}"
GPU="${GPU:-0}"
CLOCK_MHZ="${CLOCK_MHZ:-2200}"
SMA_USER="${SMA_USER:-${SUDO_USER:-}}"
SMA_REPO="${SMA_REPO:-$(cd "$HERE/../.." && pwd)}"
SMA_DISPATCHER_CONFIG="${SMA_DISPATCHER_CONFIG:-.sma/v100/sma-v100.yaml}"
INSTALL_DISPATCHER="${INSTALL_DISPATCHER:-1}"
ALLOW_DISPATCHER_CONTROL="${ALLOW_DISPATCHER_CONTROL:-0}"
SUDOERS_FILE=/etc/sudoers.d/sma-dispatcher

UNIT_DIR=/etc/systemd/system
DROPIN=50-kernelbench-isolation.conf
INSTALLED_COPY=/usr/local/sbin/kernelbench-host-setup
IRQ_RECORD=/run/kernelbench-isolation/irqs-refused
IRQBALANCE_DEFAULTS=/etc/default/irqbalance
WRITTEN_BY="# Written by services/kernelbench/host-setup.sh (SMA). Run it again to change this file."

die() { echo "host-setup.sh: $*" >&2; exit 2; }

# ---- CPU lists ("10-14,16")

cpus_expand() {  # a CPU list as space-separated CPU numbers
  local part lo hi c out=() parts=()
  IFS=, read -ra parts <<<"$1"
  for part in "${parts[@]}"; do
    lo=${part%-*}
    hi=${part#*-}
    for ((c = lo; c <= hi; c++)); do out+=("$c"); done
  done
  echo "${out[*]}"
}

cpus_compact() {  # CPU numbers as a sorted CPU list without repeats
  local c start="" prev="" out=""
  for c in $(printf '%s\n' "$@" | sort -n -u); do
    if [[ -z $start ]]; then
      start=$c
    elif ((c != prev + 1)); then
      out+="${out:+,}$start$([[ $prev == "$start" ]] || echo "-$prev")"
      start=$c
    fi
    prev=$c
  done
  if [[ -n $start ]]; then out+="${out:+,}$start$([[ $prev == "$start" ]] || echo "-$prev")"; fi
  echo "$out"
}

cpus_normal() { cpus_compact $(cpus_expand "$1"); }

cpus_overlap() {  # the CPUs two lists share, as a CPU list
  local -A in_first=()
  local c common=()
  for c in $(cpus_expand "$1"); do in_first[$c]=1; done
  for c in $(cpus_expand "$2"); do
    if [[ -n ${in_first[$c]:-} ]]; then common+=("$c"); fi
  done
  cpus_compact "${common[@]}"
}

cpus_mask() {  # a CPU list as the hex mask /proc/irq/default_smp_affinity takes (32-bit words)
  local c i top=0 words=() out=""
  for c in $(cpus_expand "$1"); do
    i=$((c / 32))
    words[i]=$((${words[i]:-0} | (1 << (c % 32))))
    if ((i > top)); then top=$i; fi
  done
  for ((i = top; i >= 0; i--)); do
    if [[ -z $out ]]; then out=$(printf '%x' "${words[i]:-0}"); else out+=$(printf ',%08x' "${words[i]:-0}"); fi
  done
  echo "$out"
}

declare -A KB_SET=()
touches_kb() {  # does the CPU list $1 name any of KB_CPUS? (no subshells: it runs once per process)
  local part lo hi c parts=()
  IFS=, read -ra parts <<<"$1"
  for part in "${parts[@]}"; do
    lo=${part%-*}
    hi=${part#*-}
    for ((c = lo; c <= hi; c++)); do
      if [[ -n ${KB_SET[$c]:-} ]]; then return 0; fi
    done
  done
  return 1
}

# ---- Sizes ("32G", base 1024 as systemd reads them)

to_bytes() {
  local n=${1%[KMGT]}
  case ${1##*[0-9]} in
    K) echo $((n << 10)) ;; M) echo $((n << 20)) ;; G) echo $((n << 30)) ;; T) echo $((n << 40)) ;; *) echo "$n" ;;
  esac
}

gib() {  # bytes as GiB with one decimal
  local tenths=$((($1 * 10 + (1 << 29)) >> 30))
  echo "$((tenths / 10)).$((tenths % 10))"
}

# ---- Parameters

check_params() {
  local name
  for name in KB_CPUS KB_TIMING_CPUS KB_COMPILE_CPUS HARNESS_CPUS; do
    [[ ${!name} =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] ||
      die "$name=${!name} is not a CPU list such as 10-19 or 10-14,16."
  done
  for name in KB_MEMORY_MIN HARNESS_MEMORY_MAX HARNESS_MEMORY_HIGH SYSTEM_MEMORY_MAX USER_MEMORY_MAX; do
    [[ ${!name} =~ ^[0-9]+[KMGT]?$ ]] || die "$name=${!name} is not a size such as 32G (a whole number, then K, M, G or T)."
  done
  [[ -z $(cpus_overlap "$KB_CPUS" "$HARNESS_CPUS") ]] ||
    die "KB_CPUS ($KB_CPUS) and HARNESS_CPUS ($HARNESS_CPUS) overlap. Give each its own CPUs."
  for name in KB_TIMING_CPUS KB_COMPILE_CPUS; do
    [[ $(cpus_overlap "${!name}" "$KB_CPUS") == "$(cpus_normal "${!name}")" ]] ||
      die "$name (${!name}) is not inside KB_CPUS ($KB_CPUS)."
  done
  [[ -z $(cpus_overlap "$KB_TIMING_CPUS" "$KB_COMPILE_CPUS") ]] ||
    die "KB_TIMING_CPUS ($KB_TIMING_CPUS) and KB_COMPILE_CPUS ($KB_COMPILE_CPUS) overlap. Compiles must stay off the timing CPUs."
  (($(to_bytes "$HARNESS_MEMORY_HIGH") <= $(to_bytes "$HARNESS_MEMORY_MAX"))) ||
    die "HARNESS_MEMORY_HIGH ($HARNESS_MEMORY_HIGH) is above HARNESS_MEMORY_MAX ($HARNESS_MEMORY_MAX)."
  [[ $GPU =~ ^[0-9A-Fa-f:.]+$ ]] || die "GPU=$GPU is not a GPU index or PCI bus id as nvidia-smi -i takes."
  [[ $CLOCK_MHZ =~ ^[0-9]+$ ]] || die "CLOCK_MHZ=$CLOCK_MHZ is not a whole number of MHz."
  [[ $INSTALL_DISPATCHER =~ ^[01]$ ]] || die "INSTALL_DISPATCHER is 1 or 0."
  [[ $ALLOW_DISPATCHER_CONTROL =~ ^[01]$ ]] || die "ALLOW_DISPATCHER_CONTROL is 1 or 0."
  local c
  for c in $(cpus_expand "$KB_CPUS"); do KB_SET[$c]=1; done
}

check_dispatcher_params() {
  [[ $INSTALL_DISPATCHER == 1 ]] || return 0
  [[ -n $SMA_USER ]] ||
    die "SMA_USER is empty. Run this with sudo from your own account, or set SMA_USER to the user the dispatcher runs as."
  [[ $SMA_USER != root ]] || die "The dispatcher must not run as root. Set SMA_USER to your own user."
  [[ $SMA_USER =~ ^[a-z_][a-z0-9_-]*$ ]] || die "SMA_USER=$SMA_USER is not a user name."
  [[ $SMA_REPO =~ ^/[A-Za-z0-9._/-]+$ ]] ||
    die "SMA_REPO=$SMA_REPO must be an absolute path of letters, digits, '.', '_', '-' and '/'."
  [[ $SMA_DISPATCHER_CONFIG =~ ^[A-Za-z0-9._/-]+$ ]] ||
    die "SMA_DISPATCHER_CONFIG=$SMA_DISPATCHER_CONFIG must be a path of letters, digits, '.', '_', '-' and '/'."
}

# ---- What apply writes

kernelbench_slice() {
  cat <<EOF
$WRITTEN_BY
# KernelBench's CPUs and memory on the harness GB10. Its container runs here
# (docker run --cgroup-parent kernelbench.slice --cpuset-cpus $KB_CPUS). MemoryMin keeps the
# container's own memory from being reclaimed for others. The GPU's allocations are charged to no
# cgroup: the caps on every other slice are what keep room for them.
[Unit]
Description=KernelBench evaluation service (CPUs $KB_CPUS, $KB_MEMORY_MIN memory kept)

[Slice]
AllowedCPUs=$KB_CPUS
MemoryMin=$KB_MEMORY_MIN
EOF
}

sma_harness_slice() {
  cat <<EOF
$WRITTEN_BY
# SMA's harness on the GB10: the dispatcher (sma-dispatcher.service) and SMA's own containers
# (docker run --cgroup-parent sma-harness.slice). It stays off KernelBench's CPUs ($KB_CPUS) and
# under its memory cap, so KernelBench keeps $KB_MEMORY_MIN.
[Unit]
Description=SMA harness (CPUs $HARNESS_CPUS, at most $HARNESS_MEMORY_MAX memory)

[Slice]
AllowedCPUs=$HARNESS_CPUS
MemoryHigh=$HARNESS_MEMORY_HIGH
MemoryMax=$HARNESS_MEMORY_MAX
EOF
}

slice_dropin() {  # $1: what runs there, $2: its memory cap
  cat <<EOF
$WRITTEN_BY
# $1 stay off KernelBench's CPUs ($KB_CPUS) and under a memory cap, so KernelBench keeps $KB_MEMORY_MIN.
[Slice]
AllowedCPUs=$HARNESS_CPUS
MemoryMax=$2
EOF
}

init_scope_dropin() {
  cat <<EOF
$WRITTEN_BY
# systemd itself stays off KernelBench's CPUs ($KB_CPUS).
[Scope]
AllowedCPUs=$HARNESS_CPUS
EOF
}

clocks_unit() {
  local shipped="$HERE/kernelbench-gpu-clocks.service"
  grep -q '^Environment=GPU=' "$shipped" || die "$shipped has no Environment=GPU= line to fill in."
  sed "s/^Environment=GPU=.*/Environment=GPU=$GPU CLOCK_MHZ=$CLOCK_MHZ/" "$shipped"
}

irq_unit() {
  cat <<EOF
$WRITTEN_BY
# Moves device interrupts off KernelBench's CPUs at every boot. IRQs the kernel manages itself
# (such as NVMe queues, which serve the CPUs they sit on) refuse to move; they are recorded in
# $IRQ_RECORD for host-setup.sh --check.
[Unit]
Description=Keep device interrupts off KernelBench's CPUs ($KB_CPUS)

[Service]
Type=oneshot
RemainAfterExit=yes
Environment=KB_CPUS=$KB_CPUS KB_TIMING_CPUS=$KB_TIMING_CPUS KB_COMPILE_CPUS=$KB_COMPILE_CPUS HARNESS_CPUS=$HARNESS_CPUS
ExecStart=$INSTALLED_COPY --irqs

[Install]
WantedBy=multi-user.target
EOF
}

dispatcher_unit() {
  local template="$HERE/sma-dispatcher.service" out
  out=$(sed -e "s|@SMA_USER@|$SMA_USER|g" -e "s|@SMA_REPO@|$SMA_REPO|g" \
    -e "s|@SMA_DISPATCHER_CONFIG@|$SMA_DISPATCHER_CONFIG|g" "$template")
  if grep -q '@[A-Z_]*@' <<<"$out"; then die "$template has a placeholder this script does not fill."; fi
  echo "$out"
}

dispatcher_sudoers() {  # the one rule ALLOW_DISPATCHER_CONTROL=1 adds: these three commands, nothing else
  local systemctl verb
  systemctl=$(command -v systemctl || echo /usr/bin/systemctl)
  echo "$WRITTEN_BY"
  echo "# Lets $SMA_USER start, stop, and restart SMA's dispatcher (sma-harness.slice) without a password."
  for verb in start stop restart; do
    echo "$SMA_USER ALL=(root) NOPASSWD: $systemctl $verb sma-dispatcher.service"
  done
}

irqbalance_installed() { command -v irqbalance >/dev/null 2>&1 || [[ -e $IRQBALANCE_DEFAULTS ]]; }

each_file() {  # $1: put (write it) or show (print it)
  local act=$1
  kernelbench_slice | $act "$UNIT_DIR/kernelbench.slice"
  sma_harness_slice | $act "$UNIT_DIR/sma-harness.slice"
  slice_dropin "System services" "$SYSTEM_MEMORY_MAX" | $act "$UNIT_DIR/system.slice.d/$DROPIN"
  slice_dropin "Login sessions" "$USER_MEMORY_MAX" | $act "$UNIT_DIR/user.slice.d/$DROPIN"
  init_scope_dropin | $act "$UNIT_DIR/init.scope.d/$DROPIN"
  clocks_unit | $act "$UNIT_DIR/kernelbench-gpu-clocks.service"
  irq_unit | $act "$UNIT_DIR/kernelbench-irq-affinity.service"
  if [[ $INSTALL_DISPATCHER == 1 ]]; then dispatcher_unit | $act "$UNIT_DIR/sma-dispatcher.service"; fi
  if [[ $INSTALL_DISPATCHER == 1 && $ALLOW_DISPATCHER_CONTROL == 1 && $act == show ]]; then
    dispatcher_sudoers | show "$SUDOERS_FILE"
  fi
}

put_sudoers() {  # the sudoers rule, checked by visudo before it is moved into place, mode 0440
  local tmp
  tmp=$(mktemp)
  dispatcher_sudoers >"$tmp"
  if ! visudo -cf "$tmp" >/dev/null; then
    rm -f "$tmp"
    echo "  visudo refused the dispatcher rule; $SUDOERS_FILE was not written"
    return 0
  fi
  install -m 0440 -o root -g root "$tmp" "$SUDOERS_FILE"
  rm -f "$tmp"
  echo "  wrote     $SUDOERS_FILE ($SMA_USER may start, stop, and restart sma-dispatcher.service)"
}

put() {  # write stdin to $1 when it differs, through a temporary file in the same folder
  local path=$1 mode=${2:-0644} tmp
  mkdir -p "$(dirname "$path")"
  tmp=$(mktemp "$path.XXXXXX")
  cat >"$tmp"
  if [[ -f $path ]] && cmp -s "$tmp" "$path"; then
    rm -f "$tmp"
    echo "  unchanged $path"
  else
    chmod "$mode" "$tmp"
    mv -f "$tmp" "$path"
    echo "  wrote     $path"
  fi
}

show() {
  echo "==> $1 <=="
  cat
  echo
}

print_all() {
  check_params
  SMA_USER="${SMA_USER:-$(id -un)}"
  check_dispatcher_params
  each_file show
  echo "==> $INSTALLED_COPY <=="
  echo "(a copy of $SELF, for kernelbench-irq-affinity.service)"
  echo
  echo "==> $IRQBALANCE_DEFAULTS (this line is set; only when irqbalance is installed) <=="
  echo "IRQBALANCE_BANNED_CPULIST=$KB_CPUS"
  echo
  echo "==> /proc/irq/default_smp_affinity <=="
  cpus_mask "$HARNESS_CPUS"
  echo
  echo "==> /proc/irq/*/smp_affinity_list (each IRQ that can reach $KB_CPUS) <=="
  echo "$HARNESS_CPUS"
  echo
  echo "==> then it runs <=="
  echo "systemctl daemon-reload"
  echo "systemctl set-property --runtime ... (the values above, on the running units)"
  echo "systemctl start kernelbench.slice sma-harness.slice"
  echo "systemctl enable kernelbench-irq-affinity.service kernelbench-gpu-clocks.service"
  echo "systemctl restart kernelbench-gpu-clocks.service (when its file changed or it is not active)"
  echo "systemctl try-restart irqbalance.service (when irqbalance is installed)"
}

# ---- Apply (root)

move_irqs() {
  local mask f n moved=0 refused=() cpus
  mask=$(cpus_mask "$HARNESS_CPUS")
  # New IRQs start on the harness CPUs too.
  if ! { echo "$mask" >/proc/irq/default_smp_affinity; } 2>/dev/null; then
    echo "  could not set /proc/irq/default_smp_affinity to $mask"
  fi
  for f in /proc/irq/[0-9]*/smp_affinity_list; do
    n=${f#/proc/irq/}
    n=${n%%/*}
    cpus=$(<"$f") || continue
    touches_kb "$cpus" || continue
    if { echo "$HARNESS_CPUS" >"$f"; } 2>/dev/null; then
      moved=$((moved + 1))
    else
      refused+=("$n")
    fi
  done
  install -d -m 0755 "$(dirname "$IRQ_RECORD")"
  printf '%s\n' "${refused[@]}" >"$IRQ_RECORD"
  chmod 0644 "$IRQ_RECORD"
  echo "  IRQs: moved $moved to CPUs $HARNESS_CPUS; ${#refused[@]} refused (per-CPU or kernel-managed), recorded in $IRQ_RECORD."
}

set_irqbalance() {
  local line="IRQBALANCE_BANNED_CPULIST=$KB_CPUS"
  if [[ -f $IRQBALANCE_DEFAULTS ]] && grep -q '^IRQBALANCE_BANNED_CPULIST=' "$IRQBALANCE_DEFAULTS"; then
    sed -i "s/^IRQBALANCE_BANNED_CPULIST=.*/$line/" "$IRQBALANCE_DEFAULTS"
  else
    printf '%s\n%s\n' "$WRITTEN_BY" "$line" >>"$IRQBALANCE_DEFAULTS"
  fi
  echo "  set       $line in $IRQBALANCE_DEFAULTS"
  systemctl try-restart irqbalance.service || echo "  irqbalance.service did not restart; check: systemctl status irqbalance"
}

check_cpus_online() {
  local online c
  online=$(</sys/devices/system/cpu/online)
  local -A up=()
  for c in $(cpus_expand "$online"); do up[$c]=1; done
  for c in $(cpus_expand "$KB_CPUS,$HARNESS_CPUS"); do
    [[ -n ${up[$c]:-} ]] || die "CPU $c is not online (online: $online). Check KB_CPUS and HARNESS_CPUS."
  done
}

apply() {
  [[ $(id -u) == 0 ]] ||
    die "Applying needs root. Run it with sudo: sudo bash $0  (--check and --print need no root)."
  command -v systemctl >/dev/null 2>&1 || die "systemctl was not found. This needs systemd."
  check_params
  check_dispatcher_params
  check_cpus_online

  local clocks_changed=1
  if clocks_unit | cmp -s - "$UNIT_DIR/kernelbench-gpu-clocks.service" 2>/dev/null &&
    systemctl is-active --quiet kernelbench-gpu-clocks.service; then
    clocks_changed=0
  fi

  echo "Files:"
  each_file put
  put "$INSTALLED_COPY" 0755 <"$SELF"
  if irqbalance_installed; then set_irqbalance; else echo "  irqbalance is not installed: nothing to ban"; fi

  echo "Units:"
  systemctl daemon-reload
  # The files take effect at boot. These set the same values on the running units now.
  systemctl set-property --runtime system.slice AllowedCPUs="$HARNESS_CPUS" MemoryMax="$SYSTEM_MEMORY_MAX"
  systemctl set-property --runtime user.slice AllowedCPUs="$HARNESS_CPUS" MemoryMax="$USER_MEMORY_MAX"
  systemctl set-property --runtime init.scope AllowedCPUs="$HARNESS_CPUS"
  systemctl start kernelbench.slice sma-harness.slice
  systemctl set-property --runtime kernelbench.slice AllowedCPUs="$KB_CPUS" MemoryMin="$KB_MEMORY_MIN"
  systemctl set-property --runtime sma-harness.slice AllowedCPUs="$HARNESS_CPUS" \
    MemoryHigh="$HARNESS_MEMORY_HIGH" MemoryMax="$HARNESS_MEMORY_MAX"
  echo "  slices: kernelbench.slice on CPUs $KB_CPUS with $KB_MEMORY_MIN kept; sma-harness.slice on CPUs $HARNESS_CPUS"
  echo "          with at most $HARNESS_MEMORY_MAX; system.slice at most $SYSTEM_MEMORY_MAX; user.slice at most $USER_MEMORY_MAX"
  systemctl enable --quiet kernelbench-irq-affinity.service
  move_irqs
  systemctl enable --quiet kernelbench-gpu-clocks.service
  if ((clocks_changed)); then
    if systemctl restart kernelbench-gpu-clocks.service; then
      echo "  GPU $GPU: clock locked at $CLOCK_MHZ MHz, compute mode EXCLUSIVE_PROCESS"
    else
      echo "  GPU $GPU: kernelbench-gpu-clocks.service failed. See: journalctl -u kernelbench-gpu-clocks.service"
    fi
  else
    echo "  GPU $GPU: kernelbench-gpu-clocks.service is active and unchanged; left running"
  fi

  if [[ $INSTALL_DISPATCHER == 1 ]]; then
    echo "Dispatcher: installed sma-dispatcher.service (user $SMA_USER, $SMA_REPO). It is not started."
    [[ -x $SMA_REPO/.venv/bin/sma ]] || echo "  $SMA_REPO/.venv/bin/sma is missing: run uv sync in $SMA_REPO first."
    [[ -f $SMA_REPO/$SMA_DISPATCHER_CONFIG || -f $SMA_DISPATCHER_CONFIG ]] ||
      echo "  $SMA_DISPATCHER_CONFIG is missing: write it before you start the dispatcher."
    if [[ $ALLOW_DISPATCHER_CONTROL == 1 ]]; then put_sudoers; fi
    echo "  Start it:   sudo systemctl enable --now sma-dispatcher.service"
    echo "  Follow it:  journalctl -u sma-dispatcher.service -f"
  fi
  echo
  echo "Check:"
  check
}

# ---- Check (no root needed)

NAMES=() OKS=() DETAILS=()
report() {  # $1: ok or fail, $2: name, $3: detail
  NAMES+=("$2")
  OKS+=("$1")
  DETAILS+=("$3")
}

unit_prop() {  # systemctl show's value, or empty when systemctl fails
  systemctl show -p "$2" --value "$1" 2>/dev/null || true
}

check_cpus_prop() {  # $1: unit, $2: wanted CPU list
  local got effective
  got=$(unit_prop "$1" AllowedCPUs)
  effective=$(unit_prop "$1" EffectiveCPUs)
  if [[ -n $got && $(cpus_normal "$got") == "$(cpus_normal "$2")" ]]; then
    report ok "$1/AllowedCPUs" "$got (effective ${effective:-none yet})"
  else
    report fail "$1/AllowedCPUs" "${got:-unset}, want $2 (effective ${effective:-unknown})"
  fi
}

check_memory_prop() {  # $1: unit, $2: property, $3: wanted size
  local got want
  got=$(unit_prop "$1" "$2")
  want=$(to_bytes "$3")
  if [[ $got == "$want" ]]; then
    report ok "$1/$2" "$(gib "$got") GiB"
  else
    report fail "$1/$2" "$([[ $got =~ ^[0-9]+$ ]] && echo "$(gib "$got") GiB" || echo "${got:-unknown}"), want $3"
  fi
}

check_slices() {
  if ! command -v systemctl >/dev/null 2>&1; then
    report fail slices "systemctl was not found"
    return
  fi
  check_cpus_prop kernelbench.slice "$KB_CPUS"
  check_memory_prop kernelbench.slice MemoryMin "$KB_MEMORY_MIN"
  check_cpus_prop sma-harness.slice "$HARNESS_CPUS"
  check_memory_prop sma-harness.slice MemoryMax "$HARNESS_MEMORY_MAX"
  check_memory_prop sma-harness.slice MemoryHigh "$HARNESS_MEMORY_HIGH"
  check_cpus_prop system.slice "$HARNESS_CPUS"
  check_memory_prop system.slice MemoryMax "$SYSTEM_MEMORY_MAX"
  check_cpus_prop user.slice "$HARNESS_CPUS"
  check_memory_prop user.slice MemoryMax "$USER_MEMORY_MAX"
  check_cpus_prop init.scope "$HARNESS_CPUS"
}

check_processes() {  # processes allowed on KB_CPUS that are not in kernelbench.slice
  local d pid key val name ppid allowed cg count=0 examples=()
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    name="" ppid="" allowed="" cg=""
    {
      while IFS=: read -r key val; do
        val=${val#"${val%%[![:space:]]*}"}
        case $key in
          Name) name=$val ;; PPid) ppid=$val ;; Cpus_allowed_list) allowed=$val && break ;;
        esac
      done <"$d/status"
      read -r cg <"$d/cgroup"
    } 2>/dev/null || continue
    # Kernel threads (kthreadd, pid 2, and its children) belong to every CPU by design.
    [[ $pid == 2 || $ppid == 2 || -z $allowed ]] && continue
    cg=${cg#0::}
    [[ $cg == /kernelbench.slice || $cg == /kernelbench.slice/* ]] && continue
    touches_kb "$allowed" || continue
    count=$((count + 1))
    if ((${#examples[@]} < 5)); then examples+=("$pid $name on $allowed in $cg"); fi
  done
  if ((count == 0)); then
    report ok processes "no process outside kernelbench.slice may run on CPUs $KB_CPUS"
  else
    local joined
    joined=$(printf '%s; ' "${examples[@]}")
    report fail processes "$count processes outside kernelbench.slice may run on CPUs $KB_CPUS: ${joined%; }"
  fi
}

check_irqs() {
  local d n cpus record=0 movable=() managed=()
  local -A refused=()
  if [[ -r $IRQ_RECORD ]]; then
    record=1
    while read -r n; do
      if [[ -n $n ]]; then refused[$n]=1; fi
    done <"$IRQ_RECORD"
  fi
  for d in /proc/irq/[0-9]*; do
    n=${d##*/}
    # The CPU an IRQ is delivered to. An empty list is a per-CPU IRQ or one not in use.
    if [[ -e $d/effective_affinity_list ]]; then
      cpus=$(<"$d/effective_affinity_list") || continue
    else
      cpus=$(<"$d/smp_affinity_list") || continue
    fi
    touches_kb "$cpus" || continue
    if [[ -n ${refused[$n]:-} ]]; then managed+=("$n"); else movable+=("$n"); fi
  done
  local note="${#managed[@]} more the kernel will not move (kernel-managed, such as NVMe queues that serve their own CPUs)"
  if ((${#movable[@]} == 0)); then
    report ok irqs "no movable IRQ reaches CPUs $KB_CPUS; $note"
  elif ((record)); then
    report fail irqs "${#movable[@]} IRQs reach CPUs $KB_CPUS (${movable[*]:0:10}); $note. Run sudo bash host-setup.sh again"
  else
    report fail irqs "${#movable[@]} IRQs reach CPUs $KB_CPUS (${movable[*]:0:10}) and no move is recorded since boot. Run sudo bash host-setup.sh"
  fi
}

check_irqbalance() {
  if ! irqbalance_installed; then
    report ok irqbalance "not installed"
    return
  fi
  local got
  got=$(sed -n 's/^IRQBALANCE_BANNED_CPULIST=//p' "$IRQBALANCE_DEFAULTS" 2>/dev/null | tail -1)
  if [[ -n $got && $(cpus_normal "$got") == "$(cpus_normal "$KB_CPUS")" ]]; then
    report ok irqbalance "IRQBALANCE_BANNED_CPULIST=$got"
  else
    report fail irqbalance "IRQBALANCE_BANNED_CPULIST is ${got:-unset} in $IRQBALANCE_DEFAULTS, want $KB_CPUS"
  fi
}

check_gpu() {
  local mode state
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    report fail gpu/compute_mode "nvidia-smi was not found"
  elif ! mode=$(timeout 30 nvidia-smi -i "$GPU" --query-gpu=compute_mode --format=csv,noheader 2>&1); then
    report fail gpu/compute_mode "nvidia-smi failed: $(head -1 <<<"$mode")"
  elif [[ $mode == Exclusive_Process ]]; then
    report ok gpu/compute_mode "GPU $GPU is Exclusive_Process"
  else
    report fail gpu/compute_mode "GPU $GPU is $mode, want Exclusive_Process"
  fi
  state=$(systemctl is-active kernelbench-gpu-clocks.service 2>/dev/null || true)
  if [[ $state == active ]]; then
    report ok gpu/clock_lock "kernelbench-gpu-clocks.service is active"
  else
    report fail gpu/clock_lock "kernelbench-gpu-clocks.service is ${state:-unknown}, want active"
  fi
}

check_memory() {
  local key val total=0 available=0 caps=0 parts="" unit got bad=()
  while read -r key val _; do
    case $key in MemTotal:) total=$((val * 1024)) ;; MemAvailable:) available=$((val * 1024)) ;; esac
  done </proc/meminfo
  for unit in sma-harness.slice system.slice user.slice; do
    got=$(unit_prop "$unit" MemoryMax)
    if [[ $got =~ ^[0-9]+$ ]]; then
      caps=$((caps + got))
      parts+="${parts:+, }${unit%.slice} $(gib "$got")"
    else
      bad+=("$unit")
    fi
  done
  local want left
  want=$(to_bytes "$KB_MEMORY_MIN")
  left=$((total - caps))
  local detail="MemTotal $(gib "$total") GiB - caps $(gib "$caps") GiB (${parts:-none}) = $(gib "$left") GiB for KernelBench (wants $(gib "$want")) and the kernel; available now $(gib "$available") GiB"
  if ((${#bad[@]})); then
    report fail memory "no MemoryMax on ${bad[*]}; $detail"
  elif ((left >= want)); then
    report ok memory "$detail"
  else
    report fail memory "$detail"
  fi
}

check_container() {
  local out parent cpuset running
  if ! command -v docker >/dev/null 2>&1; then
    report ok container "docker is not installed; skipped"
    return
  fi
  if ! timeout 10 docker version -f '{{.Server.Version}}' >/dev/null 2>&1; then
    report ok container "docker is not reachable from this user; skipped (run as a docker group member or with sudo)"
    return
  fi
  if ! out=$(timeout 10 docker container inspect -f '{{.HostConfig.CgroupParent}} {{.HostConfig.CpusetCpus}} {{.State.Running}}' \
    kernelbench 2>/dev/null); then
    report ok container "no container named kernelbench"
    return
  fi
  read -r parent cpuset running <<<"$out"
  if [[ $parent == kernelbench.slice && -n $cpuset && $(cpus_normal "$cpuset") == "$(cpus_normal "$KB_CPUS")" ]]; then
    report ok container "kernelbench is in kernelbench.slice on CPUs $cpuset (running: $running)"
  else
    report fail container "kernelbench has --cgroup-parent '${parent}' --cpuset-cpus '${cpuset}', want kernelbench.slice and $KB_CPUS (running: $running)"
  fi
}

json_str() {
  local s=${1//[[:cntrl:]]/ }
  s=${s//\\/\\\\}
  s=${s//\"/\\\"}
  printf '"%s"' "$s"
}

check() {
  check_params
  check_slices
  check_processes
  check_irqs
  check_irqbalance
  check_gpu
  check_memory
  check_container
  local i failed=0
  for i in "${!NAMES[@]}"; do
    if [[ ${OKS[i]} != ok ]]; then failed=1; fi
  done
  if ((JSON)); then
    echo "["
    for i in "${!NAMES[@]}"; do
      printf '  {"name": %s, "ok": %s, "detail": %s}%s\n' "$(json_str "${NAMES[i]}")" \
        "$([[ ${OKS[i]} == ok ]] && echo true || echo false)" "$(json_str "${DETAILS[i]}")" \
        "$( ((i + 1 < ${#NAMES[@]})) && echo , || true)"
    done
    echo "]"
  else
    for i in "${!NAMES[@]}"; do printf '%-4s %s %s\n' "${OKS[i]}" "${NAMES[i]}" "${DETAILS[i]}"; done
  fi
  return "$failed"
}

# ---- Main

MODE=apply
JSON=0
for arg in "$@"; do
  case $arg in
    --check) MODE=check ;;
    --print) MODE=print ;;
    --irqs) MODE=irqs ;;
    --json) JSON=1 ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if ((JSON)) && [[ $MODE != check ]]; then die "--json goes with --check."; fi

case $MODE in
  apply) apply ;;
  check) check ;;
  print) print_all ;;
  irqs)
    [[ $(id -u) == 0 ]] || die "--irqs needs root."
    check_params
    move_irqs
    ;;
esac
