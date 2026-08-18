import fs from 'node:fs';
import path from 'node:path';

const roots = ['src', 'web-tests'];
const issues = [];

function walk(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(directory, entry.name);
    return entry.isDirectory() ? walk(full) : [full];
  });
}

for (const file of roots.flatMap(walk).filter((file) => /\.(js|jsx)$/.test(file))) {
  const source = fs.readFileSync(file, 'utf8');
  if (/console\.(log|debug)\(/.test(source)) issues.push(`${file}: remove debug console output`);
  if (/dangerouslySetInnerHTML/.test(source)) issues.push(`${file}: avoid dangerouslySetInnerHTML`);
  if (/\t/.test(source)) issues.push(`${file}: tabs are not used in workbench source`);
  if (/\s+$/.test(source.split('\n').find((line) => /\s+$/.test(line)) ?? '')) issues.push(`${file}: trailing whitespace`);
}

if (issues.length) {
  console.error(issues.join('\n'));
  process.exit(1);
}
console.log('Workbench source hygiene passed.');
