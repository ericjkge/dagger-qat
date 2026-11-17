# server helpers (no sim env here: sim processes source simenv.sh only, mesa software EGL)
unset CUDA_VISIBLE_DEVICES; COMPAT=$(nvidia-smi | grep -q "CUDA Version: 12" && echo /usr/local/cuda-13.0/compat:); export LD_LIBRARY_PATH=${COMPAT}/usr/local/cuda-13.0/lib64
stop_server() { pkill -f "[s]glang.launch_server"; for i in $(seq 1 60); do pgrep -f "[s]glang.launch_server" >/dev/null || break; sleep 2; done; sleep 5; }
start_server() {
  stop_server; rm -f /workspace/logs/server_A.log; setsid /workspace/serve.sh A /workspace/models/pear "$@" < /dev/null > /dev/null 2>&1 &
  for i in $(seq 1 300); do
    curl -s -m 3 127.0.0.1:30000/get_model_info | grep -q model_path && return 0
    grep -q "kill_process_tree\|Traceback" /workspace/logs/server_A.log 2>/dev/null && { echo "SERVER_FAILED"; return 1; }
    sleep 6
  done; echo "SERVER_TIMEOUT"; return 1
}
