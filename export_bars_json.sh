#!/usr/bin/env bash
set -Eeuo pipefail

# Export SAT bars as JSON array: [{ts, open, high, low, close, volume}, ...]
# Usage:
#   ./export_bars_json.sh <TICKER> <TIMEFRAME> <RANGE> <OUT.json>
# Optional envs:
#   COMPOSE_SVC=db
#   DB_USER=satuser
#   DB_NAME=sat
#   VERBOSE=1

trap 'echo "❌ Failed (line $LINENO). See messages above." >&2' ERR

TICKER="${1:-ETHUSDT}"
TF="${2:-30m}"
DRANGE="${3:-2y}"
OUT="${4:-./bars.json}"

COMPOSE_SVC="${COMPOSE_SVC:-db}"
DB_USER="${DB_USER:-satuser}"
DB_NAME="${DB_NAME:-sat}"
VERBOSE="${VERBOSE:-}"

if [[ -z "$TICKER" || -z "$TF" || -z "$DRANGE" || -z "$OUT" ]]; then
  echo "Usage: $0 <TICKER> <TIMEFRAME> <RANGE> <OUT.json>" >&2
  exit 1
fi

# Escape single quotes for SQL literals
esc() { printf "%s" "$1" | sed "s/'/''/g"; }
TQ="$(esc "$TICKER")"
TFQ="$(esc "$TF")"
DRQ="$(esc "$DRANGE")"

# SQL template (filled below). Always emits a single row JSON array.
SQL_TEMPLATE="$(cat <<'SQL_EOF'
SELECT COALESCE(
  (
    SELECT json_agg(
             json_build_object(
               'ts',     (bar->>'ts')::bigint,
               'open',   COALESCE((bar->>'open')::double precision,  (bar->>'o')::double precision),
               'high',   COALESCE((bar->>'high')::double precision,  (bar->>'h')::double precision),
               'low',    COALESCE((bar->>'low')::double precision,   (bar->>'l')::double precision),
               'close',  COALESCE((bar->>'close')::double precision, (bar->>'c')::double precision),
               'volume', COALESCE((bar->>'volume')::double precision,(bar->>'v')::double precision, 0)
             )
             ORDER BY (bar->>'ts')::bigint
           )
    FROM (
      SELECT data_json
      FROM price
      WHERE ticker = '__TICKER__'
        AND timeframe = '__TF__'
        AND "range" = '__DRANGE__'
      ORDER BY price_id DESC
      LIMIT 1
    ) lr
    CROSS JOIN LATERAL jsonb_array_elements(lr.data_json) AS bar
  ),
  '[]'::json
);
SQL_EOF
)"

SQL="${SQL_TEMPLATE//'__TICKER__'/$TQ}"
SQL="${SQL//'__TF__'/$TFQ}"
SQL="${SQL//'__DRANGE__'/$DRQ}"

if [[ -n "$VERBOSE" ]]; then
  echo "Service    : $COMPOSE_SVC"
  echo "DB         : $DB_NAME as $DB_USER"
  echo "Ticker     : $TICKER"
  echo "Timeframe  : $TF"
  echo "Range      : $DRANGE"
  echo "Output     : $OUT"
  echo "----- SQL -----"
  echo "$SQL"
  echo "---------------"
fi

mkdir -p "$(dirname "$OUT")"
TMP="${OUT}.tmp"

# Run the query inside the db container
docker compose exec -T "$COMPOSE_SVC" \
  psql -U "$DB_USER" -d "$DB_NAME" \
  -v ON_ERROR_STOP=1 -At --pset footer=off \
  -c "$SQL" > "$TMP"

# Pretty-print if jq exists, else keep compact JSON
if command -v jq >/dev/null 2>&1; then
  jq '.' "$TMP" > "$OUT"
  rm -f "$TMP"
else
  mv "$TMP" "$OUT"
fi

# Sanity checks
if [[ ! -s "$OUT" ]]; then
  echo "⚠️  Output file is empty: $OUT" >&2
  exit 1
fi
first="$(head -c1 "$OUT" || true)"
if [[ "$first" != "[" && "$first" != "{" ]]; then
  echo "⚠️  Output doesn't look like JSON (first char: '$first')." >&2
  [[ -n "$VERBOSE" ]] && sed -n '1,3p' "$OUT" >&2
  exit 1
fi

echo "✅ Wrote JSON to: $OUT"

