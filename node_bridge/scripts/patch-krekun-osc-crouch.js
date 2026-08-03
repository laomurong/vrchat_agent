// Patch @iflow-mcp/krekun-vrchat-mcp-osc: install the shared pressHoldRelease()
// helper and fix the Moveleft typo (run via npm postinstall, pure ASCII).
//
// History / why crouch & prone were REMOVED from this patch:
//   This script used to also register `crouch` and `prone` MCP tools that sent
//   /input/Crouch and /input/Prone. That never worked: VRChat's official OSC
//   input whitelist (Buttons + Axes) does NOT contain Crouch or Prone -- C and Z
//   are desktop-client-only local key bindings with no OSC address. Because OSC
//   runs over UDP there is no receipt, so the bridge happily reported success
//   while VRChat silently dropped the packet ("whole chain green, avatar never
//   crouches"). Shipping those tools only produced a convincing illusion.
//
//   Crouch / prone are now driven from the Python side by real keyboard
//   injection (Win32 PostMessage -> VRChat window), see
//   src/vrchat_agent/action/key_sender.py and the `send_key` pseudo tool in
//   mappings/avatar_default.yaml. Nothing OSC-related remains here for poses.
//
// What this patch still does (both are load-bearing, do not drop):
//   1. inputTools.js: fix typo 'Moveleft' -> 'MoveLeft' (VRChat OSC input names
//      are case-sensitive, so /input/Moveleft never worked for move:left).
//   2. inputTools.js: add the shared `pressHoldRelease(inputName, duration, label)`
//      helper (press 1.0 -> hold -> release 0.0, with safety timeout and retry).
//      patch-krekun-osc-expand.js has a HARD dependency on this helper: all 11 of
//      its button tools delegate to it and its main() aborts when it is missing.
//
// Idempotent: if the pressHoldRelease marker is already present in inputTools.js,
// the patch is skipped (safe to re-run on every npm install / postinstall).
//
// Note: uses string replacement on dist files with stable anchors; if upstream
// changes the surrounding code, the patch reports "anchor not found" and exits
// with a clear message instead of silently corrupting the file.
'use strict';

const fs = require('fs');
const path = require('path');

const PKG_ROOT = path.join(
  __dirname, '..', 'node_modules', '@iflow-mcp', 'krekun-vrchat-mcp-osc'
);
const SERVER_PATH = path.join(PKG_ROOT, 'packages', 'mcp-server', 'dist', 'server.js');
const INPUT_TOOLS_PATH = path.join(PKG_ROOT, 'packages', 'mcp-server', 'dist', 'tools', 'inputTools.js');

// Marker that proves this patch already ran (checked for idempotency).
// Lives in inputTools.js next to the helper it guards -- server.js is no longer
// touched by this patch since crouch/prone tool registration was removed.
const MARKER = '// [patch-krekun-osc-crouch] pressHoldRelease helper installed';

/**
 * Patch a single file with an anchor-based replacement.
 * Throws a descriptive Error when the anchor is missing (prevents silent corruption).
 * @param {string} filePath absolute path to the file to patch
 * @param {string} anchor unique substring that must exist before the replacement
 * @param {string} replacement new content to insert (or replace the target block)
 * @param {boolean} insertBefore whether to insert replacement before the anchor
 */
function patchFile(filePath, anchor, replacement, insertBefore) {
  if (!fs.existsSync(filePath)) {
    throw new Error('file not found: ' + filePath);
  }
  let src = fs.readFileSync(filePath, 'utf8');
  const idx = src.indexOf(anchor);
  if (idx === -1) {
    throw new Error(
      '[patch-krekun-osc-crouch] anchor not found in ' + filePath +
      ' (upstream changed?): ' + anchor.slice(0, 60)
    );
  }
  if (insertBefore) {
    src = src.slice(0, idx) + replacement + src.slice(idx);
  } else {
    src = src.slice(0, idx) + replacement + src.slice(idx + anchor.length);
  }
  fs.writeFileSync(filePath, src, 'utf8');
  console.log('[patch-krekun-osc-crouch] patched: ' + filePath);
}

/**
 * Replace all occurrences of a simple literal (used for the Moveleft typo fix).
 * @param {string} filePath absolute path to the file
 * @param {string} from exact literal to replace
 * @param {string} to replacement literal
 * @returns {number} number of replacements made
 */
function replaceAll(filePath, from, to) {
  if (!fs.existsSync(filePath)) {
    throw new Error('file not found: ' + filePath);
  }
  const src = fs.readFileSync(filePath, 'utf8');
  if (src.indexOf(from) === -1) {
    return 0;
  }
  const out = src.split(from).join(to);
  fs.writeFileSync(filePath, out, 'utf8');
  return (src.match(new RegExp(from.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')) || []).length;
}

function main() {
  if (!fs.existsSync(SERVER_PATH) || !fs.existsSync(INPUT_TOOLS_PATH)) {
    console.warn('[patch-krekun-osc-crouch] package dist files not found, skip');
    process.exit(0);
  }
  const toolsSrc = fs.readFileSync(INPUT_TOOLS_PATH, 'utf8');
  if (toolsSrc.indexOf(MARKER) !== -1) {
    console.log('[patch-krekun-osc-crouch] already patched (pressHoldRelease present), skip');
    process.exit(0);
  }

  // ---- 1) inputTools.js: fix Moveleft -> MoveLeft typo ----
  const typoCount = replaceAll(INPUT_TOOLS_PATH, "inputName = 'Moveleft';", "inputName = 'MoveLeft'; // [patch-krekun-osc-crouch] typo fix");
  if (typoCount === 0) {
    // Not fatal: maybe upstream already fixed it; log and continue.
    console.log('[patch-krekun-osc-crouch] Moveleft typo not found (already fixed upstream?), skip');
  } else {
    console.log('[patch-krekun-osc-crouch] fixed Moveleft -> MoveLeft (' + typoCount + ' occurrence)');
  }

  // ---- 2) inputTools.js: add the shared pressHoldRelease helper ----
  // Anchor: the JSDoc comment before the existing look() method (stable marker
  // right after move() ends). Insert the helper before it.
  //
  // NOTE: crouch()/prone() methods used to be inserted here too. They are gone
  // on purpose (no OSC address exists for them, see the file header); the helper
  // stays because patch-krekun-osc-expand.js's button tools all delegate to it.
  const lookAnchor = '    /**\n     * Turn to look in a specific direction.';
  const newMethods = `    ${MARKER}
    /**
     * Press an input, hold for duration seconds, then release.
     * Shared by every button-style VRChat input (Run / UseRight / GrabLeft /
     * ToggleSitStand / ... registered by patch-krekun-osc-expand.js), modeled on
     * move()'s press-hold-release pattern with a safety timeout.
     *
     * @param inputName - VRChat OSC input name (e.g. 'Run', 'ToggleSitStand')
     * @param duration - Hold duration in seconds
     * @param label - Human-readable label for log/message text
     * @returns Promise resolving to a confirmation message
     */
    async pressHoldRelease(inputName, duration, label) {
        let releaseTimer = null;
        const ensureRelease = async () => {
            for (let attempt = 1; attempt <= 3; attempt++) {
                try {
                    const ok = await this.wsClient.sendInput(inputName, 0.0);
                    if (ok) {
                        return true;
                    }
                    if (attempt < 3) {
                        await delay(100);
                    }
                }
                catch (error) {
                    if (attempt < 3) {
                        await delay(200);
                    }
                }
            }
            return false;
        };
        try {
            releaseTimer = setTimeout(async () => {
                logger.warn(\`Safety timeout for \${inputName} triggered\`);
                await ensureRelease();
            }, (duration + 0.5) * 1000);
            let pressSuccess = false;
            for (let attempt = 1; attempt <= 3; attempt++) {
                try {
                    pressSuccess = await this.wsClient.sendInput(inputName, 1.0);
                    if (pressSuccess) {
                        break;
                    }
                    if (attempt < 3) {
                        await delay(100);
                    }
                }
                catch (error) {
                    if (attempt < 3) {
                        await delay(200);
                    }
                }
            }
            if (!pressSuccess) {
                if (releaseTimer) {
                    clearTimeout(releaseTimer);
                    releaseTimer = null;
                }
                await ensureRelease();
                return \`Failed to start \${label}\`;
            }
            await delay(duration * 1000);
            const releaseSuccess = await ensureRelease();
            if (releaseTimer) {
                clearTimeout(releaseTimer);
                releaseTimer = null;
            }
            if (!releaseSuccess) {
                return \`Started \${label} but failed to stop completely\`;
            }
            return \`\${label.charAt(0).toUpperCase() + label.slice(1)} for \${duration} seconds\`;
        }
        catch (error) {
            const errorMsg = \`Error during \${label}: \${error instanceof Error ? error.message : String(error)}\`;
            logger.error(errorMsg);
            if (releaseTimer) {
                clearTimeout(releaseTimer);
            }
            try {
                await this.wsClient.sendInput(inputName, 0.0);
                logger.info(\`Emergency release of \${inputName} after error\`);
            }
            catch (releaseError) {
                logger.error(\`Failed emergency release: \${releaseError instanceof Error ? releaseError.message : String(releaseError)}\`);
            }
            return errorMsg;
        }
    }
`;
  patchFile(INPUT_TOOLS_PATH, lookAnchor, newMethods, true);

  // ---- 3) server.js: intentionally untouched ----
  // crouch/prone MCP tools were removed (VRChat has no /input/Crouch|Prone).
  // Poses are handled by Python-side keyboard injection; see the file header.

  console.log('[patch-krekun-osc-crouch] pressHoldRelease helper + MoveLeft typo fix applied (idempotent marker written)');
}

main();
