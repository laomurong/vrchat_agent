// Fix @iflow-mcp/krekun-vrchat-mcp-osc publish defect (run via npm postinstall).
//
// Background: this package is published from a pnpm monorepo, but the tarball
// omits two things that the compiled dist files need at runtime:
//
// 1. Relative imports to pkg-root utils/types.
//    packages/mcp-server/dist/server.js and submodules import
//    '../../../utils/dist/index.js' and '../../../types/dist/index.js'
//    (pkg-root utils/dist, types/dist), but the tarball only contains
//    packages/* directories -> ERR_MODULE_NOT_FOUND at startup.
//
// 2. Workspace package alias @vrchat-mcp-osc/utils.
//    packages/relay-server/dist/index.js (spawned by mcp-server) imports
//    '@vrchat-mcp-osc/utils' which never exists on the npm registry.
//    -> ERR_MODULE_NOT_FOUND when the relay server starts.
//
// Fix:
//   - copy packages/utils/dist and packages/types/dist into the package root
//   - create a virtual package at node_modules/@vrchat-mcp-osc/utils that
//     points at packages/utils/dist, so ESM resolution of the workspace name
//     succeeds inside the relay server.
//
// Idempotent: re-runs overwrite with latest content.
//
// Note: uses readdirSync/copyFileSync instead of fs.cpSync because cpSync
// crashes the Node process on some Windows/sandbox setups.
'use strict';

const fs = require('fs');
const path = require('path');

const NODE_MODULES = path.join(__dirname, '..', 'node_modules');
const PKG_ROOT = path.join(NODE_MODULES, '@iflow-mcp', 'krekun-vrchat-mcp-osc');

/**
 * Recursively copy a directory tree (files, subdirectories, symlinks).
 * @param {string} srcDir source directory absolute path
 * @param {string} destDir destination directory absolute path
 */
function copyDirRecursive(srcDir, destDir) {
  fs.mkdirSync(destDir, { recursive: true });
  const entries = fs.readdirSync(srcDir, { withFileTypes: true });
  for (const entry of entries) {
    const srcPath = path.join(srcDir, entry.name);
    const destPath = path.join(destDir, entry.name);
    if (entry.isDirectory()) {
      copyDirRecursive(srcPath, destPath);
    } else if (entry.isFile()) {
      fs.copyFileSync(srcPath, destPath);
    } else if (entry.isSymbolicLink()) {
      const target = fs.readlinkSync(srcPath);
      fs.symlinkSync(target, destPath);
    }
  }
}

/**
 * Copy a directory into destDir (create parent dirs as needed).
 * @param {string} srcDir source directory absolute path
 * @param {string} destDir destination directory absolute path
 */
function syncDir(srcDir, destDir) {
  if (!fs.existsSync(srcDir)) {
    console.log('[patch-krekun-utils] skip (source missing): ' + srcDir);
    return;
  }
  copyDirRecursive(srcDir, destDir);
  console.log('[patch-krekun-utils] synced: ' + srcDir + ' -> ' + destDir);
}

/**
 * Create a virtual scoped package directory that aliases a workspace package.
 * @param {string} scopeDir e.g. node_modules/@vrchat-mcp-osc
 * @param {string} pkgName e.g. utils
 * @param {string} srcDistDir dist source inside the published package
 */
function installWorkspaceAlias(scopeDir, pkgName, srcDistDir) {
  const pkgDir = path.join(scopeDir, pkgName);
  fs.mkdirSync(path.join(pkgDir, 'dist'), { recursive: true });
  copyDirRecursive(srcDistDir, path.join(pkgDir, 'dist'));
  const pkgJson = {
    name: '@vrchat-mcp-osc/' + pkgName,
    version: '0.1.0',
    main: 'dist/index.js',
    module: 'dist/index.js',
    types: 'dist/index.d.ts',
    type: 'module',
    dependencies: { winston: '^3.11.0' },
  };
  fs.writeFileSync(
    path.join(pkgDir, 'package.json'),
    JSON.stringify(pkgJson, null, 2),
    'utf8'
  );
  console.log('[patch-krekun-utils] workspace alias created: ' + pkgDir);
}

function main() {
  const pkgJson = path.join(PKG_ROOT, 'package.json');
  if (!fs.existsSync(pkgJson)) {
    console.warn('[patch-krekun-utils] package @iflow-mcp/krekun-vrchat-mcp-osc not found, skip');
    process.exit(0);
  }
  // Fix 1: pkg-root utils/types for mcp-server relative imports.
  syncDir(path.join(PKG_ROOT, 'packages', 'utils', 'dist'), path.join(PKG_ROOT, 'utils', 'dist'));
  syncDir(path.join(PKG_ROOT, 'packages', 'types', 'dist'), path.join(PKG_ROOT, 'types', 'dist'));
  // Fix 2: @vrchat-mcp-osc/utils workspace alias for the relay server.
  installWorkspaceAlias(
    path.join(NODE_MODULES, '@vrchat-mcp-osc'),
    'utils',
    path.join(PKG_ROOT, 'packages', 'utils', 'dist')
  );
  console.log('[patch-krekun-utils] krekun-vrchat-mcp-osc patched');
}

main();
