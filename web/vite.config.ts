import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Proxies API calls to the FastAPI backend so the dev server never has to
// deal with CORS - see `uv run uvicorn api.main:app --app-dir src --reload`.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/chat': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/metrics': 'http://localhost:8000',
      '/cache': 'http://localhost:8000',
      '/knowledge': 'http://localhost:8000',
    },
  },
});
