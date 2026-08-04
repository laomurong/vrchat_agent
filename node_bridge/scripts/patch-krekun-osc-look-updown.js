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
// Idempotent: each target file carries its OWN marker, written only after
// that file's patches all succeed (A-20). If a later step fails, re-running
// converges instead of skipping the whole patch on a half-applied state.
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

// Markers that prove this patch already ran (checked per file for idempotency).
var MARKER_SERVER = '// [patch-krekun-osc-look-updown] server.js look up/down directions registered';
var MARKER_TOOLS = '// [patch-krekun-osc-look-updown] inputTools.js look up/down directions registered';
// Legacy marker from the pre-A-20 script: it was written into server.js only.
// Treat it as "server.js already patched" so old half-applied states converge.
var MARKER_SERVER_LEGACY = '// [patch-krekun-osc-look-updown] look up/down directions registered';

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

/**
 * Replace a single anchor, treating "already replaced" as success (converges).
 *
 * A-20: an anchor may be missing either because upstream changed it OR because
 * a previous (old-script / half-applied) run already replaced it. If the target
 * text is already present we consider the step done; only when neither the
 * anchor nor the target text exists do we fail loudly.
 * @param {string} filePath absolute path to the file
 * @param {string} from exact literal to replace
 * @param {string} to replacement literal
 * @param {string} label human-readable step name for log/error messages
 * @returns {boolean} true if the step was applied or already present
 */
function replaceConverge(filePath, from, to, label) {
  var src = fs.readFileSync(filePath, 'utf8');
  if (src.indexOf(from) !== -1) {
    var out = src.split(from).join(to);
    fs.writeFileSync(filePath, out, 'utf8');
    console.log('[patch-krekun-osc-look-updown] ' + label + ' patched');
    return true;
  }
  if (src.indexOf(to) !== -1) {
    console.log('[patch-krekun-osc-look-updown] ' + label + ' already present, skip');
    return true;
  }
  throw new Error(
    '[patch-krekun-osc-look-updown] ' + label + ' anchor not found ' +
    '(upstream changed?): ' + from.slice(0, 80)
  );
}

/**
 * Append a marker comment to the end of a file (idempotent).
 * @param {string} filePath absolute path to the file
 * @param {string} marker marker text to append
 */
function appendMarker(filePath, marker) {
  var src = fs.readFileSync(filePath, 'utf8');
  if (src.indexOf(marker) !== -1) {
    return;
  }
  fs.writeFileSync(filePath, src + '\n' + marker + '\n', 'utf8');
}

function main() {
  if (!fs.existsSync(SERVER_PATH)) {
    console.warn('[patch-krekun-osc-look-updown] package server.js not found, skip');
    process.exit(0);
  }

  // ---- 1) server.js: extend look_direction enum to include up/down ----
  var serverSrc = fs.readFileSync(SERVER_PATH, 'utf8');
  if (serverSrc.indexOf(MARKER_SERVER) !== -1 || serverSrc.indexOf(MARKER_SERVER_LEGACY) !== -1) {
    console.log('[patch-krekun-osc-look-updown] server.js already patched, skip');
  } else {
    var enumFrom = "direction: z.enum(['left', 'right']).describe('Direction to look'),";
    var enumTo = "direction: z.enum(['left', 'right', 'up', 'down']).describe('Direction to look'),";
    replaceConverge(SERVER_PATH, enumFrom, enumTo, 'server.js look_direction enum');
    // Marker written ONLY after server.js patches succeed.
    appendMarker(SERVER_PATH, MARKER_SERVER);
  }

  // ---- 2) inputTools.js: relax direction validation in look() ----
  var checkFrom = "if (direction !== 'left' && direction !== 'right') {";
  var checkTo = "if (!['left', 'right', 'up', 'down'].includes(direction)) {";

  // ---- 3) inputTools.js: replace ternary inputName with a map including up/down ----
  var inputNameFrom = "const inputName = direction === 'left' ? 'LookLeft' : 'LookRight';";
  var inputNameTo = "const inputNameMap = { left: 'LookLeft', right: 'LookRight', up: 'LookUp', down: 'LookDown' };\n        const inputName = inputNameMap[direction];";

  var toolsSrc = fs.readFileSync(INPUT_TOOLS_PATH, 'utf8');
  if (toolsSrc.indexOf(MARKER_TOOLS) !== -1) {
    console.log('[patch-krekun-osc-look-updown] inputTools.js already patched, skip');
  } else {
    replaceConverge(INPUT_TOOLS_PATH, checkFrom, checkTo, 'inputTools.js direction validation');
    replaceConverge(INPUT_TOOLS_PATH, inputNameFrom, inputNameTo, 'inputTools.js inputName map');
    // Marker written ONLY after BOTH inputTools.js steps succeed.
    appendMarker(INPUT_TOOLS_PATH, MARKER_TOOLS);
  }

  console.log('[patch-krekun-osc-look-updown] all patches applied (per-file idempotent markers written)');
}

main();
