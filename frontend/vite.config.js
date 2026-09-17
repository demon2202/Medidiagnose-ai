import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  resolve: {
    // Never bundle a second copy of React into optimized deps
    // (framer-motion + React duplication breaks hooks at runtime).
    dedupe: ['react', 'react-dom'],
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    // Allow cloud preview hosts (suffix match) + local dev.
    allowedHosts: ['.e2b.app', 'localhost'],
  },
});
