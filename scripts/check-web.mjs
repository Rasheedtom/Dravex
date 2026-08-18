import fs from 'node:fs';

const required = [
  'package.json',
  'package-lock.json',
  'index.html',
  'src/main.jsx',
  'src/App.jsx',
  'vite.config.js',
];

for (const file of required) {
  if (!fs.existsSync(file)) {
    throw new Error(`Required workbench file is missing: ${file}`);
  }
}

const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
if (pkg.engines?.node !== '>=20 <21') throw new Error('Workbench must stay pinned to Node 20.');
if (!pkg.scripts?.test || !pkg.scripts?.build) throw new Error('Workbench test/build scripts are required.');

const lockfiles = ['package-lock.json', 'pnpm-lock.yaml', 'yarn.lock'].filter((file) => fs.existsSync(file));
if (lockfiles.length !== 1) throw new Error(`Expected exactly one JavaScript lockfile, found ${lockfiles.length}.`);
console.log('Workbench repository checks passed.');
