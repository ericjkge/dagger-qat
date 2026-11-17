#!/usr/bin/env bash
# Stops (not terminates) this pod at a hard deadline, or after 60 consecutive minutes with no work process.
# Uses the pod-scoped key via GraphQL podStop (runpodctl's REST path is unauthorized with that key).
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | xargs)
DEADLINE=$(date -u -d "${DEADLINE_UTC:-2026-11-15 05:00}" +%s)
stop_pod() {
  echo "$(date -u) $1 stop" >> /workspace/watchdog.log
  curl -s -H 'Content-Type: application/json' \
    -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}" \
    "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" >> /workspace/watchdog.log
  echo >> /workspace/watchdog.log
}
idle=0
while true; do
  now=$(date -u +%s)
  [ "$now" -ge "$DEADLINE" ] && stop_pod deadline
  if pgrep -f "[p]ython|[s]glang|hf download|[h]f_transfer|uv pip" >/dev/null; then idle=0; else idle=$((idle+1)); fi
  [ "$idle" -ge 60 ] && stop_pod idle
  echo "$(date -u) idle=$idle deadline=$(date -u -d @$DEADLINE)" > /workspace/watchdog.status
  sleep 60
done
