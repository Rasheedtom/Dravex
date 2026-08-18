import fs from 'node:fs';
import { describe, expect, it } from 'vitest';

describe('repository contract', () => {
  it('uses Node 20 and one frozen npm lockfile', () => {
    const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
    expect(pkg.engines.node).toBe('>=20 <21');
    expect(fs.existsSync('package-lock.json')).toBe(true);
    expect(fs.existsSync('pnpm-lock.yaml')).toBe(false);
    expect(fs.existsSync('yarn.lock')).toBe(false);
  });

  it('keeps the existing Python package alongside the workbench', () => {
    expect(fs.existsSync('pyproject.toml')).toBe(true);
    expect(fs.existsSync('dravex/core/signal.py')).toBe(true);
  });
});
