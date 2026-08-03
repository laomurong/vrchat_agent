// Patch @iflow-mcp/krekun-vrchat-mcp-osc to add up/down look directions.
//
// Background: the look_direction tool only accepts direction enum
// ['left', 'right']. VRChat OSC inputs are generic (sendInput sends
// /input/<inputName>), so LookUp / LookDown can be driven the same way
// as LookLeft / LookRight. This patch extends the enum and the input
// name mapping in two files:
//
//   1. server.js: extend the look_direction tool's z.enum to include
//      'up' and 'down' (so the MCP server accepts them).
//   2. inputTools.js: relax the direction validation in look() to accept
//      up/down, and replace the ternary inputName assignment with a map
//      that includes LookUp / LookDown.
//
// Idempotent: if the marker comment is already present in server.js,
// the patch is skipped (safe to re-run on every npm install / postinstall).
//
// Note: uses string replacement on dist files with stable anchors; if
// upstream changes the surrounding code, the patch reports "anchor not
// found" and exits with a clear message instead of silently corrupting
// the file.
'use strict';

var fs = require('fs');
var path = require('path');

var PKG_ROOT = path.join(
  __dirname, '..', 'node_modules', '@iflow-mcp', 'krekun-vrchat-mcp-osc'
);
var SERVER_PATH = path.join(PKG_ROOT, 'packages', 'mcp-server', 'dist', 'server.js');
var INPUT_TOOLS_PATH = path.join(PKG_ROOT, 'packages', 'mcp-server', 'dist', 'tools', 'inputTools.js');

// Marker that proves this patch already ran (checked for idempotency).
var MARKER = '// [patch-krekun-osc-look-updown] look up/down directions registered';

/**
 * Replace all occurrences of a simple literal in a file.
 * @param {string} filePath absolute path to the file
 * @param {string} from exact literal to replace
 * @param {string} to replacement literal
 * @returns {number} number of replacements made
 */
function replaceAll(filePath, from, to) {
  if (!fs.existsSync(filePath)) {
    throw new Error('file not found: ' + filePath);
  }
  var src = fs.readFileSync(filePath, 'utf8');
  if (src.indexOf(from) === -1) {
    return 0;
  }
  var out = src.split(from).join(to);
  fs.writeFileSync(filePath, out, 'utf8');
  return src.split(from).length - 1;
}

function main() {
  if (!fs.existsSync(SERVER_PATH)) {
    console.warn('[patch-krekun-osc-look-updown] package server.js not found, skip');
    process.exit(0);
  }
  var serverSrc = fs.readFileSync(SERVER_PATH, 'utf8');
  if (serverSrc.indexOf(MARKER) !== -1) {
    console.log('[patch-krekun-osc-look-updown] already patched (up/down present), skip');
    process.exit(0);
  }

  // ---- 1) server.js: extend look_direction enum to include up/down ----
  var enumFrom = "direction: z.enum(['left', 'right']).describe('Direction to look'),";
  var enumTo = "direction: z.enum(['left', 'right', 'up', 'down']).describe('Direction to look'), " + MARKER;
  var enumCount = replaceAll(SERVER_PATH, enumFrom, enumTo);
  if (enumCount === 0) {
    throw new Error(
      '[patch-krekun-osc-look-updown] server.js enum anchor not found ' +
      '(upstream changed?): ' + enumFrom
    );
  }
  console.log('[patch-krekun-osc-look-updown] server.js: look_direction enum extended (up/down), ' + enumCount + ' replacement');

  // ---- 2) inputTools.js: relax direction validation in look() ----
  var checkFrom = "if (direction !== 'left' && direction !== 'right') {";
  var checkTo = "if (!['left', 'right', 'up', 'down'].includes(direction)) {";
  var checkCount = replaceAll(INPUT_TOOLS_PATH, checkFrom, checkTo);
  if (checkCount === 0) {
    throw new Error(
      '[patch-krekun-osc-look-updown] inputTools.js direction check anchor not found ' +
      '(upstream changed?): ' + checkFrom
    );
  }
  console.log('[patch-krekun-osc-look-updown] inputTools.js: direction validation relaxed, ' + checkCount + ' replacement');

  // ---- 3) inputTools.js: replace ternary inputName with a map including up/down ----
  var inputNameFrom = "const inputName = direction === 'left' ? 'LookLeft' : 'LookRight';";
  var inputNameTo = "const inputNameMap = { left: 'LookLeft', right: 'LookRight', up: 'LookUp', down: 'LookDown' };\n        const inputName = inputNameMap[direction];";
  var inputNameCount = replaceAll(INPUT_TOOLS_PATH, inputNameFrom, inputNameTo);
  if (inputNameCount === 0) {
    throw new Error(
      '[patch-krekun-osc-look-updown] inputTools.js inputName anchor not found ' +
      '(upstream changed?): ' + inputNameFrom
    );
  }
  console.log('[patch-krekun-osc-look-updown] inputTools.js: inputName map extended (LookUp/LookDown), ' + inputNameCount + ' replacement');

  console.log('[patch-krekun-osc-look-updown] all patches applied (idempotent marker written)');
}

main();
