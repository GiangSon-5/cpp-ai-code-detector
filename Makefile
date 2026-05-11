dev:
	@chmod +x dev.sh
	@./dev.sh

stop:
	@tmux kill-session -t lvtn_dev 2>/dev/null || echo "No session running"
	@pkill -f "uvicorn src.fastapi_service.main:app" || true
	@pkill -f "python manage.py runserver" || true
	@pkill -f "celery -A src.celery_workers.celery_app" || true
	@pkill -f "uvicorn src.local_gpu_server.main:app" || true

logs:
	@tmux attach-session -t lvtn_dev
