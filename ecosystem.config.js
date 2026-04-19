module.exports = {
  apps: [
    {
      name: "hallmark-api",
      cwd: "/home/ubuntu/ocr-detection",
      script: "venv/bin/uvicorn",
      args: "api:app --host 0.0.0.0 --port 8000",
      interpreter: "none",
      env: {
        PATH: "/home/ubuntu/ocr-detection/venv/bin:/usr/local/bin:/usr/bin:/bin",
        PYTHONPATH: "/home/ubuntu/ocr-detection",
      },
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      log_date_format: "YYYY-MM-DD HH:mm:ss Z",
      error_file: "/home/ubuntu/ocr-detection/logs/api-error.log",
      out_file: "/home/ubuntu/ocr-detection/logs/api-out.log",
      merge_logs: true,
    },
  ],
};
