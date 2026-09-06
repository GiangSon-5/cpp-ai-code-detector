#!/bin/bash

# =================================================================
# CPP AI DETECTOR AUTOMATION SCRIPT
# =================================================================

SESSION_NAME="cpp_detector_dev"
CONDA_ENV="detection_ai"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Kiểm tra nếu đang chạy bên trong một session tmux khác
if [ -n "$TMUX" ]; then
    echo "⚠️ Bạn đang ở trong một session tmux. Hãy nhấn Ctrl+B rồi D để thoát ra trước khi chạy lệnh này."
    exit 1
fi

# Tắt session cũ nếu có
tmux kill-session -t $SESSION_NAME 2>/dev/null
sleep 1 # Đợi server dọn dẹp

# Tạo session mới và không kết nối ngay (-d)
tmux new-session -d -s $SESSION_NAME -n 'Services' -c $PROJECT_DIR

# Đảm bảo tmux server đã sẵn sàng
sleep 0.5

# Hàm để chạy lệnh trong một pane cụ thể
run_in_pane() {
    local pane_id=$1
    local cmd=$2
    local label=$3
    
    # Kích hoạt Conda environment linh hoạt trên nhiều cấu hình máy
    tmux send-keys -t $SESSION_NAME:0.$pane_id "echo '>>> $label' && (source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true) && conda activate $CONDA_ENV 2>/dev/null; $cmd" C-m
}

# Chia màn hình (Layout 2x2)
tmux split-window -h -t $SESSION_NAME:0.0 -c $PROJECT_DIR
tmux split-window -v -t $SESSION_NAME:0.0 -c $PROJECT_DIR
tmux split-window -v -t $SESSION_NAME:0.1 -c $PROJECT_DIR

# Khởi chạy các service
run_in_pane 0 "uvicorn src.fastapi_service.main:app --port 8001" "AI Orchestrator (8001)"
run_in_pane 1 "python manage.py runserver 8000" "Web Frontend (8000)"
run_in_pane 2 "celery -A src.celery_workers.celery_app worker --loglevel=info" "Celery Worker"
run_in_pane 3 "uvicorn src.local_gpu_server.main:app --host 0.0.0.0 --port 8002" "Local GPU Worker (8002)"

# Cân bằng layout
tmux select-layout -t $SESSION_NAME:0 tiled

# Thông báo
echo "✅ Đã khởi chạy các service trong session tmux: $SESSION_NAME"
echo "👉 Đang kết nối vào màn hình quản lý..."
sleep 1

# Kết nối
tmux attach-session -t $SESSION_NAME
