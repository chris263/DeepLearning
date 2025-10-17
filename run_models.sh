#!/usr/bin/env bash
# ./run_models.sh --5 --py /usr/local/bin/python3.11 --ticker BTCUSDT --timeframe 1h
# Adds: --ticker, --timeframe (default: ETHUSDT, 4h)

set -u

SCRIPT="${SCRIPT:-./cnn_ichimoku_strategy.py}"

# Fixed params (kept from your exact command; change here if you like)
LOOKBACK=64
HORIZON=6
POS_THR=0.004
EPOCHS=15
BATCH=256
PROB_THRESHOLD=0.55
DSN_DEFAULT='postgresql://postgres:postgres@localhost:5432/sat'

# Parse args
RUNS=10
PY=""
DSN="$DSN_DEFAULT"
TICKER="ETHUSDT"
TIMEFRAME="4h"

# sugar: --5
if [[ $# -gt 0 && "$1" =~ ^--[0-9]+$ ]]; then RUNS="${1#--}"; shift; fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs|-n)     shift; RUNS="${1:?}"; shift;;
    --py)          shift; PY="${1:?}"; shift;;
    --script)      shift; SCRIPT="${1:?}"; shift;;
    --dsn)         shift; DSN="${1:?}"; shift;;
    --ticker|-t)   shift; TICKER="${1:?}"; shift;;
    --timeframe|-f)shift; TIMEFRAME="${1:?}"; shift;;
    --)            shift; break;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done

# --- pick a Python that can import torch ---
has_torch () {
  command -v "$1" >/dev/null 2>&1 || return 1
  "$1" - <<'PY' >/dev/null 2>&1 || return 1
try:
  import torch  # noqa
except Exception:
  raise SystemExit(1)
PY
}
pick_py () {
  if [[ -n "$PY" ]]; then
    has_torch "$PY" && { echo "$PY"; return; }
    echo "ERROR: --py '$PY' found but cannot import torch there." >&2
    exit 127
  fi
  for cand in python3.11 python python3; do
    command -v "$cand" >/dev/null 2>&1 || continue
    has_torch "$cand" && { echo "$cand"; return; }
  done
  echo "ERROR: No Python on PATH can import torch. Use --py /path/to/python-with-torch" >&2
  exit 127
}
PYBIN="$(pick_py)"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="runs_${STAMP}"
LOGDIR="$OUTDIR/logs"
CSV="$OUTDIR/results.csv"
mkdir -p "$LOGDIR"

echo "Using Python : $(command -v "$PYBIN")"
echo "Script       : $SCRIPT"
echo "Runs         : $RUNS"
echo "Ticker/TF    : $TICKER / $TIMEFRAME"
echo "Logs         : $LOGDIR"
echo "CSV          : $CSV"

# CSV header (matches your columns)
echo "trial,seed,lookback,horizon,pos_thr,epochs,batch,prob_threshold,net_profit_pct,max_drawdown_pct,final_equity,Symbol/TF,Train period,Test period,Train samples,Test samples,Trades (closed),Wins,Losses,log_file" > "$CSV"

# Helpers
strip_ansi() { perl -pe 's/\x1B\[[0-9;]*[A-Za-z]//g'; }

extract_json_line() {
  local file="$1"
  awk '
    {
      gsub(/\x1B\[[0-9;]*[A-Za-z]/,"",$0);
      if ($0 ~ /^METRICS_JSON_OOS:/ || $0 ~ /^METRICS_JSON_IS:/ || $0 ~ /^METRICS_JSON:/) {
        sub(/^[^:]+:[[:space:]]*/,"",$0);
        print $0; exit
      }
    }
  ' "$file"
}

parse_text_block() {
  local file="$1"
  local txt; txt="$(strip_ansi < "$file")"
  local net mdd eq symtf trainp testp trains tests trades wins losses
  net="$( printf "%s" "$txt" | grep -E 'Net Profit %'   | tail -n1 | grep -Eo '[-+]?[0-9]+(\.[0-9]+)?%?' | sed 's/%$//' || true)"
  mdd="$( printf "%s" "$txt" | grep -E 'Max Drawdown %' | tail -n1 | grep -Eo '[-+]?[0-9]+(\.[0-9]+)?%?' | sed 's/%$//' || true)"
  eq="$(  printf "%s" "$txt" | grep -E 'Final Equity'   | tail -n1 | grep -Eo '[-+]?[0-9]+(\.[0-9]+)?'   || true)"
  symtf="$( printf "%s" "$txt" | grep -E '^Symbol/TF[[:space:]]*:'    | tail -n1 | sed 's/.*:[[:space:]]*//' || true)"
  trainp="$(printf "%s" "$txt" | grep -E '^Train period[[:space:]]*:' | tail -n1 | sed 's/.*:[[:space:]]*//' || true)"
  testp="$( printf "%s" "$txt" | grep -E '^Test period[[:space:]]*:'  | tail -n1 | sed 's/.*:[[:space:]]*//' || true)"
  trains="$(printf "%s" "$txt" | grep -E 'Train samples' | tail -n1 | grep -Eo '[0-9]+' | sed -n '1p' || true)"
  tests="$( printf "%s" "$txt" | grep -E 'Train samples' | tail -n1 | grep -Eo '[0-9]+' | sed -n '2p' || true)"
  trades="$(printf "%s" "$txt" | grep -E '^Trades[[:space:]]*\(closed\)' | tail -n1 | grep -Eo '[-+]?[0-9]+' || true)"
  wins="$(  printf "%s" "$txt" | grep -E '^Wins[[:space:]]*/[[:space:]]*Losses' | tail -n1 | awk -F'[:/]' '{gsub(/ /,""); print $3}' || true)"
  losses="$(printf "%s" "$txt" | grep -E '^Wins[[:space:]]*/[[:space:]]*Losses' | tail -n1 | awk -F'[:/]' '{gsub(/ /,""); print $4}' || true)"
  echo "${net:-}|${mdd:-}|${eq:-}|${symtf:-}|${trainp:-}|${testp:-}|${trains:-}|${tests:-}|${trades:-}|${wins:-}|${losses:-}"
}

# Build the exact command once (with dynamic ticker/timeframe)
CMD=( "$PYBIN" "$SCRIPT"
  --dsn "$DSN"
  --ticker "$TICKER"
  --timeframe "$TIMEFRAME"
  --lookback "$LOOKBACK"
  --horizon "$HORIZON"
  --pos_thr "$POS_THR"
  --epochs "$EPOCHS"
  --batch "$BATCH"
  --prob_threshold "$PROB_THRESHOLD"
  --commission 0.1
  --slippage_bps 3
  --device cpu
  --verbose
)

echo "Probing once..."
echo "CMD: ${CMD[*]}"
LOG1="$LOGDIR/trial_1.log"
"${CMD[@]}" >"$LOG1" 2>&1
rc=$?
echo "[trial 1] exit_code=$rc | log: $LOG1"

# Parse metrics from trial 1
json="$(extract_json_line "$LOG1" || true)"
if [[ -n "$json" ]]; then
  readarray -t vals < <(
    "$PYBIN" - <<'PY' 2>/dev/null <<<"$json"
import sys, json, re
try: d=json.load(sys.stdin)
except: sys.exit(0)
def num(x):
    if x is None: return ""
    if isinstance(x,(int,float)): return x
    s=str(x).strip()
    s=re.sub(r'%$','',s)
    try: return float(s)
    except: return ""
fields=["net_profit_pct","max_drawdown_pct","final_equity",
        "symbol_tf","train_period","test_period",
        "train_samples","test_samples","trades_closed","wins","losses"]
out=[]
for k in fields:
    v=d.get(k,"")
    if k in ("net_profit_pct","max_drawdown_pct","final_equity"):
        v=num(v)
    out.append("" if v is None else v)
print("\n".join(map(str,out)))
PY
  )
  NET="${vals[0]:-}"; MDD="${vals[1]:-}"; FEQ="${vals[2]:-}"
  SYMTF="${vals[3]:-}"; TRP="${vals[4]:-}"; TEP="${vals[5]:-}"
  TRAINS="${vals[6]:-}"; TESTS="${vals[7]:-}"; TRADES="${vals[8]:-}"
  WINS="${vals[9]:-}"; LOSSES="${vals[10]:-}"
else
  IFS='|' read -r NET MDD FEQ SYMTF TRP TEP TRAINS TESTS TRADES WINS LOSSES < <(parse_text_block "$LOG1")
fi
[[ -z "${NET:-}" ]] && NET="NaN"
[[ -z "${MDD:-}" ]] && MDD="NaN"
[[ -z "${FEQ:-}" ]] && FEQ="NaN"

echo "1,1,$LOOKBACK,$HORIZON,$POS_THR,$EPOCHS,$BATCH,$PROB_THRESHOLD,$NET,$MDD,$FEQ,\"${SYMTF:-}\",\"${TRP:-}\",\"${TEP:-}\",${TRAINS:-},${TESTS:-},${TRADES:-},${WINS:-},${LOSSES:-},$LOG1" >> "$CSV"
[[ $rc -ne 0 ]] && { echo "Probe FAILED. (See $LOG1)"; exit $rc; }

# Remaining runs (2..N)
if (( RUNS > 1 )); then
  for i in $(seq 2 "$RUNS"); do
    LOG="$LOGDIR/trial_${i}.log"
    echo "[trial $i] starting…"
    "${CMD[@]}" >"$LOG" 2>&1
    rc=$?
    echo "[trial $i] exit_code=$rc | log: $LOG"

    json="$(extract_json_line "$LOG" || true)"
    if [[ -n "$json" ]]; then
      readarray -t vals < <(
        "$PYBIN" - <<'PY' 2>/dev/null <<<"$json"
import sys, json, re
try: d=json.load(sys.stdin)
except: sys.exit(0)
def num(x):
    if x is None: return ""
    if isinstance(x,(int,float)): return x
    s=str(x).strip()
    s=re.sub(r'%$','',s)
    try: return float(s)
    except: return ""
fields=["net_profit_pct","max_drawdown_pct","final_equity",
        "symbol_tf","train_period","test_period",
        "train_samples","test_samples","trades_closed","wins","losses"]
out=[]
for k in fields:
    v=d.get(k,"")
    if k in ("net_profit_pct","max_drawdown_pct","final_equity"):
        v=num(v)
    out.append("" if v is None else v)
print("\n".join(map(str,out)))
PY
      )
      NET="${vals[0]:-}"; MDD="${vals[1]:-}"; FEQ="${vals[2]:-}"
      SYMTF="${vals[3]:-}"; TRP="${vals[4]:-}"; TEP="${vals[5]:-}"
      TRAINS="${vals[6]:-}"; TESTS="${vals[7]:-}"; TRADES="${vals[8]:-}"
      WINS="${vals[9]:-}"; LOSSES="${vals[10]:-}"
    else
      IFS='|' read -r NET MDD FEQ SYMTF TRP TEP TRAINS TESTS TRADES WINS LOSSES < <(parse_text_block "$LOG")
    fi
    [[ -z "${NET:-}" ]] && NET="NaN"
    [[ -z "${MDD:-}" ]] && MDD="NaN"
    [[ -z "${FEQ:-}" ]] && FEQ="NaN"

    echo "$i,$i,$LOOKBACK,$HORIZON,$POS_THR,$EPOCHS,$BATCH,$PROB_THRESHOLD,$NET,$MDD,$FEQ,\"${SYMTF:-}\",\"${TRP:-}\",\"${TEP:-}\",${TRAINS:-},${TESTS:-},${TRADES:-},${WINS:-},${LOSSES:-},$LOG" >> "$CSV"

    [[ $rc -ne 0 ]] && { echo "[trial $i] FAILED (see $LOG)"; exit $rc; }
  done
fi

echo
echo "All runs complete."
echo "CSV saved at: $CSV"
echo "Logs in      : $LOGDIR"
