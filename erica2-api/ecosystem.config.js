// ERICA API - PM2 Ecosystem Configuration
// Domain: https://erica.ivf20.app

module.exports = {
  apps: [
    // ============================================
    // Production
    // ============================================
    {
      name: 'erica-prod',
      script: 'uvicorn',
      args: 'main:app --host 0.0.0.0 --port 8000',
      cwd: __dirname,
      interpreter: 'python3',
      env: {
        ERICA_ENV: 'production'
      },
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '2G',
      log_file: './logs/pm2_production.log',
      error_file: './logs/pm2_production_error.log',
      merge_logs: true,
      time: true,
      max_restarts: 10,
      restart_delay: 1000
    },

    // ============================================
    // Staging
    // ============================================
    {
      name: 'erica-staging',
      script: 'uvicorn',
      args: 'main:app --host 0.0.0.0 --port 8002',
      cwd: __dirname,
      interpreter: 'python3',
      env: {
        ERICA_ENV: 'staging'
      },
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      log_file: './logs/pm2_staging.log',
      error_file: './logs/pm2_staging_error.log',
      merge_logs: true,
      time: true
    },

    // ============================================
    // Development
    // ============================================
    {
      name: 'erica-dev',
      script: 'uvicorn',
      args: 'main:app --host 0.0.0.0 --port 8001 --reload',
      cwd: __dirname,
      interpreter: 'python3',
      env: {
        ERICA_ENV: 'development'
      },
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: ['*.py', 'utils/*.py'],
      ignore_watch: ['logs', 'models', '__pycache__', 'temp_images'],
      log_file: './logs/pm2_development.log',
      error_file: './logs/pm2_development_error.log',
      merge_logs: true,
      time: true
    }
  ]
};

/*
Usage:

# Start all
pm2 start ecosystem.config.js

# Start specific
pm2 start ecosystem.config.js --only erica-prod

# Restart
pm2 restart erica-prod

# Logs
pm2 logs erica-prod

# Monitor
pm2 monit

# Save configuration
pm2 save

# Startup (auto-start on reboot)
pm2 startup
*/
