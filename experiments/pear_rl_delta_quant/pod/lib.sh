. /workspace/simenv.sh; export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
unset CUDA_VISIBLE_DEVICES; COMPAT=$(nvidia-smi | grep -q "CUDA Version: 12" && echo /usr/local/cuda-13.0/compat:); export LD_LIBRARY_PATH=${COMPAT}/usr/local/cuda-13.0/lib64
armdir() { case $1 in A) echo /workspace/models/pear;; *) echo /workspace/models/arm_$1;; esac; }
stop_server() { pkill -f "[s]glang.launch_server"; for i in $(seq 1 60); do pgrep -f "[s]glang.launch_server" >/dev/null || break; sleep 2; done; sleep 5; }
start_server() {
  stop_server; rm -f /workspace/logs/server_$1.log; setsid /workspace/serve.sh $1 $(armdir $1) < /dev/null > /dev/null 2>&1 &
  for i in $(seq 1 300); do
    curl -s -m 3 127.0.0.1:30000/get_model_info | grep -q model_path && return 0
    grep -q "kill_process_tree\|Traceback" /workspace/logs/server_$1.log 2>/dev/null && { echo "SERVER_FAILED $1"; return 1; }
    sleep 6
  done; echo "SERVER_TIMEOUT $1"; return 1
}
