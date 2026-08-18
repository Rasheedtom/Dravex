import { defineConfig } from 'vite';

export default defineConfig({
  esbuild: {
    jsx: 'automatic',
  },
  test: {
    include: ['web-tests/**/*.test.{js,jsx}'],
    environment: 'node',
    restoreMocks: true,
  },
});
