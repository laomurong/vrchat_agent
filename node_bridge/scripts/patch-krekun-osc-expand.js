// Patch @iflow-mcp/krekun-vrchat-mcp-osc to expand the OSC action surface
// (run via npm postinstall, pure ASCII).
//
// Background: upstream 0.1.3 only registers 14 MCP tools (move/look/jump/menu/
// voice/send_message/crouch/prone/avatar-*). VRChat's OSC input surface is far
// larger (see https://docs.vrchat.com/docs/osc-as-input-controller), and the
// bridge already ships two generic escape hatches that are never exposed:
//
//   * wsClient.send_message(address, value) -> relay 'sendOsc' -> oscClient
//     .send_message(address, value): can emit ANY OSC address, not just /input/*.
//   * wsClient.setAvatarParameter(name, value) -> relay 'avatar/setParameter'
//     -> /avatar/parameters/<name>.
//
// This patch layers the missing high-level actions on top of those primitives:
//
//   1. inputTools.js: add button methods (run/useRight/useLeft/grabRight/
//      grabLeft/dropRight/dropLeft/toggleSitStand/afkToggle/panic/
//      quickMenuRight) that all delegate to the existing pressHoldRelease()
//      helper added by patch-krekun-osc-crouch.js, plus:
//        - holdAxis()           shared analog-axis helper (send value, hold, zero)
//        - setMoveAxis()        /input/Vertical | /input/Horizontal
//        - setLookAxis()        /input/LookHorizontal | /input/LookVertical
//        - setGesture()         /avatar/parameters/GestureLeft | GestureRight
//        - setTypingIndicator() /chatbox/typing
//        - sendRawOsc()         arbitrary OSC address (generic escape hatch)
//   2. server.js: register the matching MCP tools using the exact same
//      registration style as the upstream look_direction / move_avatar tools.
//
// Why axes matter: /input/MoveForward etc. are binary buttons (full speed or
// nothing). /input/Vertical and /input/Horizontal are floats in [-1.0, 1.0],
// so the agent can walk slowly (0.4) instead of sprinting. VRChat requires
// axes to be returned to 0.0 when finished, which holdAxis() guarantees with
// the same safety-timeout + retry discipline pressHoldRelease() uses.
//
// Dependency: this patch MUST run AFTER patch-krekun-osc-crouch.js, because it
// reuses that patch's pressHoldRelease() helper. main() verifies this and exits
// with a clear message instead of producing a broken file.
//
// Idempotent: each target file carries its own marker; a file that already
// contains its marker is skipped, so re-running (npm install / postinstall)
// is a no-op and partially-applied states still converge.
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

// Markers that prove this patch already ran (checked per file for idempotency).
const MARKER_TOOLS = '// [patch-krekun-osc-expand] extended input methods';
const MARKER_SERVER = '// [patch-krekun-osc-expand] extended OSC tools registered';

// ---------------------------------------------------------------------------
// Button table: VRChat /input/<name> buttons driven by pressHoldRelease().
// `input` values are the EXACT VRChat OSC input names (case-sensitive!).
// ---------------------------------------------------------------------------
const BUTTONS = [
  {
    method: 'run', input: 'Run', tool: 'run', def: 1.0, label: 'run',
    desc: 'Hold the Run modifier (sprint) for the given duration.'
  },
  {
    method: 'useRight', input: 'UseRight', tool: 'use_right', def: 0.2, label: 'use right',
    desc: 'Use / interact with the right hand (equivalent to right click).'
  },
  {
    method: 'useLeft', input: 'UseLeft', tool: 'use_left', def: 0.2, label: 'use left',
    desc: 'Use / interact with the left hand.'
  },
  {
    method: 'grabRight', input: 'GrabRight', tool: 'grab_right', def: 0.5, label: 'grab right',
    desc: 'Grab an object with the right hand (hold to keep holding it).'
  },
  {
    method: 'grabLeft', input: 'GrabLeft', tool: 'grab_left', def: 0.5, label: 'grab left',
    desc: 'Grab an object with the left hand (hold to keep holding it).'
  },
  {
    method: 'dropRight', input: 'DropRight', tool: 'drop_right', def: 0.15, label: 'drop right',
    desc: 'Drop whatever the right hand is holding.'
  },
  {
    method: 'dropLeft', input: 'DropLeft', tool: 'drop_left', def: 0.15, label: 'drop left',
    desc: 'Drop whatever the left hand is holding.'
  },
  {
    method: 'toggleSitStand', input: 'ToggleSitStand', tool: 'toggle_sit_stand', def: 0.2,
    label: 'toggle sit/stand',
    desc: 'Toggle between sitting and standing on a seat / station.'
  },
  {
    method: 'afkToggle', input: 'AFKToggle', tool: 'afk_toggle', def: 0.2, label: 'toggle AFK',
    desc: 'Toggle the AFK (away from keyboard) state.'
  },
  {
    method: 'panic', input: 'PanicButton', tool: 'panic_button', def: 0.2, label: 'panic',
    desc: 'Trigger the VRChat panic button (forces Safety Mode on) - use sparingly.'
  },
  {
    method: 'quickMenuRight', input: 'QuickMenuToggleRight', tool: 'quick_menu_right', def: 0.2,
    label: 'quick menu right',
    desc: 'Toggle the right-hand quick menu.'
  }
];

/**
 * Insert `replacement` immediately before `anchor` inside `filePath`.
 * Throws a descriptive Error when the anchor is missing (prevents silent corruption).
 * @param {string} filePath absolute path to the file to patch
 * @param {string} anchor unique substring that must exist in the file
 * @param {string} replacement content inserted before the anchor
 * @returns {void}
 */
function insertBefore(filePath, anchor, replacement) {
  if (!fs.existsSync(filePath)) {
    throw new Error('file not found: ' + filePath);
  }
  const src = fs.readFileSync(filePath, 'utf8');
  const idx = src.indexOf(anchor);
  if (idx === -1) {
    throw new Error(
      '[patch-krekun-osc-expand] anchor not found in ' + filePath +
      ' (upstream changed?): ' + anchor.slice(0, 70)
    );
  }
  const out = src.slice(0, idx) + replacement + src.slice(idx);
  fs.writeFileSync(filePath, out, 'utf8');
  console.log('[patch-krekun-osc-expand] patched: ' + filePath);
}

/**
 * Build the InputTools button methods source (one method per BUTTONS entry).
 * @returns {string} JavaScript source to splice into the class body
 */
function buildButtonMethods() {
  return BUTTONS.map(function (b) {
    return `    /**
     * ${b.desc}
     *
     * Sends /input/${b.input} = 1, holds, then releases with 0.
     *
     * @param duration - Duration in seconds to hold the input
     * @param ctx - MCP Context (optional)
     * @returns Promise resolving to a confirmation message
     */
    async ${b.method}(duration = ${b.def}, ctx) {
        if (ctx) {
            await ctx.info(\`Input ${b.label} for \${duration} seconds\`);
        }
        return this.pressHoldRelease('${b.input}', duration, '${b.label}');
    }
`;
  }).join('');
}

/**
 * Build the InputTools axis / gesture / typing / raw-OSC methods source.
 * @returns {string} JavaScript source to splice into the class body
 */
function buildExtraMethods() {
  return `    /**
     * Drive an analog VRChat axis input: send \`value\`, hold for \`duration\`
     * seconds, then reset the axis to 0.0.
     *
     * VRChat requires axis inputs to be returned to 0.0 once the motion is
     * finished, otherwise the avatar keeps moving forever. This helper mirrors
     * pressHoldRelease()'s discipline: a safety timeout that zeroes the axis
     * even if the normal flow throws, plus a 3-attempt retry on the reset.
     *
     * @param inputName - VRChat axis input name (e.g. 'Vertical', 'LookHorizontal')
     * @param value - Axis value, clamped into [-1.0, 1.0]
     * @param duration - Hold duration in seconds before resetting to 0
     * @param label - Human-readable label for log/message text
     * @returns Promise resolving to a confirmation message
     */
    async holdAxis(inputName, value, duration, label) {
        const raw = Number(value);
        const axisValue = Math.max(-1, Math.min(1, Number.isFinite(raw) ? raw : 0));
        const holdSeconds = Number.isFinite(Number(duration)) && Number(duration) > 0
            ? Number(duration)
            : 0.5;
        let releaseTimer = null;
        const resetAxis = async () => {
            for (let attempt = 1; attempt <= 3; attempt++) {
                try {
                    // Typed float is REQUIRED here: VRChat axis inputs are float32,
                    // and node-osc's Message.append() downgrades a bare integer to
                    // int32 (typetag ',i'), which VRChat silently ignores - the
                    // avatar would never stop moving. The {type:'float'} form forces
                    // node-osc to emit typetag ',f' (float32) and passes opaquely
                    // through the WS JSON serialization.
                    const ok = await this.wsClient.sendInput(inputName, { type: 'float', value: 0 });
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
                logger.warn(\`Safety timeout for axis \${inputName} triggered\`);
                await resetAxis();
            }, (holdSeconds + 0.5) * 1000);
            const pressSuccess = await this.wsClient.sendInput(inputName, { type: 'float', value: axisValue });
            if (!pressSuccess) {
                if (releaseTimer) {
                    clearTimeout(releaseTimer);
                    releaseTimer = null;
                }
                await resetAxis();
                return \`Failed to set axis \${inputName}\`;
            }
            await delay(holdSeconds * 1000);
            const resetOk = await resetAxis();
            if (releaseTimer) {
                clearTimeout(releaseTimer);
                releaseTimer = null;
            }
            if (!resetOk) {
                return \`Axis \${inputName}=\${axisValue} held but failed to reset to 0\`;
            }
            return \`\${label} (\${inputName}=\${axisValue}) for \${holdSeconds}s, axis reset to 0\`;
        }
        catch (error) {
            const errorMsg = \`Error during \${label}: \${error instanceof Error ? error.message : String(error)}\`;
            logger.error(errorMsg);
            if (releaseTimer) {
                clearTimeout(releaseTimer);
            }
            try {
                await this.wsClient.sendInput(inputName, { type: 'float', value: 0 });
                logger.info(\`Emergency reset of axis \${inputName} after error\`);
            }
            catch (resetError) {
                logger.error(\`Failed emergency axis reset: \${resetError instanceof Error ? resetError.message : String(resetError)}\`);
            }
            return errorMsg;
        }
    }
    /**
     * Analog movement via /input/Vertical (forward/back) or /input/Horizontal
     * (strafe right/left). Unlike move(), which presses a binary MoveForward
     * style button, this supports partial speed (e.g. 0.4 = walk slowly).
     *
     * With run = true the /input/Run modifier is held for the whole movement
     * window (sprint). Run is a MODIFIER in VRChat: pressing it on its own is a
     * no-op, it only has an effect while a movement axis/button is active. The
     * orchestration layer above is strictly sequential (every tool call is
     * awaited to completion), so Run and Move can never overlap if they are two
     * separate calls - the concurrency has to live inside this single call.
     *
     * @param axis - 'vertical' (forward/back) or 'horizontal' (strafe)
     * @param value - Axis value in [-1.0, 1.0]; positive = forward / right
     * @param duration - Hold duration in seconds before resetting to 0
     * @param run - Hold the Run modifier for the duration of the movement
     * @param ctx - MCP Context (optional)
     * @returns Promise resolving to a confirmation message
     */
    async setMoveAxis(axis, value, duration = 0.5, run = false, ctx) {
        const axisMap = { vertical: 'Vertical', horizontal: 'Horizontal' };
        const inputName = axisMap[axis];
        if (!inputName) {
            return \`Unknown move axis: \${axis} (expected 'vertical' or 'horizontal')\`;
        }
        if (ctx) {
            await ctx.info(\`Move axis \${axis}=\${value} for \${duration} seconds\${run ? ' (running)' : ''}\`);
        }
        if (!run) {
            return this.holdAxis(inputName, value, duration, \`Move \${axis}\`);
        }
        // ---- Run modifier held across the whole movement window ----
        // Run is an int32 button, exactly like every other /input/* button, so
        // it goes through sendInput() with bare 1.0/0.0 (NOT the {type:'float'}
        // form used for axes). Release uses the same 3-attempt retry discipline
        // as pressHoldRelease.ensureRelease() in patch-krekun-osc-crouch.js.
        const holdSeconds = Number.isFinite(Number(duration)) && Number(duration) > 0
            ? Number(duration)
            : 0.5;
        const releaseRun = async () => {
            for (let attempt = 1; attempt <= 3; attempt++) {
                try {
                    const ok = await this.wsClient.sendInput('Run', 0.0);
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
        // Safety net: if holdAxis somehow never returns (hung WS call), Run must
        // still be released or the avatar sprints forever. Budget is duration +
        // 1.0s, i.e. strictly looser than holdAxis' own (duration + 0.5s) timer.
        let runTimer = setTimeout(async () => {
            logger.warn('Safety timeout for Run modifier triggered');
            await releaseRun();
        }, (holdSeconds + 1.0) * 1000);
        const clearRunTimer = () => {
            if (runTimer) {
                clearTimeout(runTimer);
                runTimer = null;
            }
        };
        let pressed = false;
        try {
            pressed = await this.wsClient.sendInput('Run', 1.0);
            if (!pressed) {
                logger.warn('Failed to press Run modifier, falling back to walking');
            }
            const result = await this.holdAxis(inputName, value, duration, \`Run \${axis}\`);
            return pressed
                ? result
                : \`\${result} (Run modifier failed to engage - walked instead of sprinted)\`;
        }
        catch (error) {
            const errorMsg = \`Error during run move \${axis}: \${error instanceof Error ? error.message : String(error)}\`;
            logger.error(errorMsg);
            return errorMsg;
        }
        finally {
            // Always release Run, even when holdAxis threw or the press failed
            // (a failed press may still have landed on the VRChat side).
            clearRunTimer();
            const released = await releaseRun();
            if (!released) {
                logger.error('Failed to release Run modifier after 3 attempts');
            }
        }
    }
    /**
     * Analog turning via /input/LookHorizontal (yaw) or /input/LookVertical
     * (pitch). Unlike look(), which taps a LookLeft/LookRight button, this gives
     * fine-grained control over turn speed via the axis magnitude.
     *
     * @param axis - 'horizontal' (yaw / turn) or 'vertical' (pitch / tilt)
     * @param value - Axis value in [-1.0, 1.0]; positive = right / up
     * @param duration - Hold duration in seconds before resetting to 0
     * @param ctx - MCP Context (optional)
     * @returns Promise resolving to a confirmation message
     */
    async setLookAxis(axis, value, duration = 0.3, ctx) {
        const axisMap = { horizontal: 'LookHorizontal', vertical: 'LookVertical' };
        const inputName = axisMap[axis];
        if (!inputName) {
            return \`Unknown look axis: \${axis} (expected 'horizontal' or 'vertical')\`;
        }
        if (ctx) {
            await ctx.info(\`Look axis \${axis}=\${value} for \${duration} seconds\`);
        }
        return this.holdAxis(inputName, value, duration, \`Look \${axis}\`);
    }
    /**
     * Set a standard VRChat hand gesture by writing the built-in avatar
     * parameter GestureLeft / GestureRight (/avatar/parameters/<name>).
     *
     * Standard VRChat gesture indices (Int 0-7):
     *   0 Neutral, 1 Fist, 2 HandOpen, 3 FingerPoint,
     *   4 Victory, 5 RockNRoll, 6 HandGun, 7 ThumbsUp
     * The gesture persists until it is set back to 0 (Neutral).
     *
     * @param side - 'left' or 'right'
     * @param gesture - Gesture index, clamped/rounded into 0..7
     * @param ctx - MCP Context (optional)
     * @returns Promise resolving to a confirmation message
     */
    async setGesture(side, gesture, ctx) {
        const paramMap = { left: 'GestureLeft', right: 'GestureRight' };
        const parameterName = paramMap[side];
        if (!parameterName) {
            return \`Unknown gesture side: \${side} (expected 'left' or 'right')\`;
        }
        const raw = Number(gesture);
        const index = Math.max(0, Math.min(7, Math.round(Number.isFinite(raw) ? raw : 0)));
        if (ctx) {
            await ctx.info(\`Setting \${parameterName} = \${index}\`);
        }
        try {
            const success = await this.wsClient.setAvatarParameter(parameterName, index);
            if (!success) {
                return \`Failed to set \${parameterName} = \${index}\`;
            }
            return \`Set \${parameterName} = \${index}\`;
        }
        catch (error) {
            const errorMsg = \`Error setting \${parameterName}: \${error instanceof Error ? error.message : String(error)}\`;
            logger.error(errorMsg);
            return errorMsg;
        }
    }
    /**
     * Toggle the VRChat chatbox "typing" indicator (/chatbox/typing).
     *
     * Note: relay-server's oscClient exposes set_typing_indicator(), but the
     * relay never routes a 'chatbox/typing' action to it, so that method is
     * unreachable dead code. We therefore go through the generic sendOsc path
     * (wsClient.send_message), which emits the exact same OSC packet.
     *
     * @param typing - true to show the typing indicator, false to hide it
     * @param ctx - MCP Context (optional)
     * @returns Promise resolving to a confirmation message
     */
    async setTypingIndicator(typing, ctx) {
        const isTyping = Boolean(typing);
        if (ctx) {
            await ctx.info(\`Setting typing indicator: \${isTyping}\`);
        }
        try {
            const success = await this.wsClient.send_message('/chatbox/typing', isTyping);
            if (!success) {
                return \`Failed to set typing indicator to \${isTyping}\`;
            }
            return \`Typing indicator set to \${isTyping}\`;
        }
        catch (error) {
            const errorMsg = \`Error setting typing indicator: \${error instanceof Error ? error.message : String(error)}\`;
            logger.error(errorMsg);
            return errorMsg;
        }
    }
    /**
     * Send a raw OSC message to an arbitrary address (generic escape hatch).
     *
     * Useful for avatar-specific parameters and any VRChat OSC address that has
     * no dedicated tool yet. Accepts the three OSC argument types the relay can
     * carry end-to-end (verified against node-osc's Message.append()):
     *
     *   number  -> OSC int32 when integral, float32 when fractional
     *   boolean -> OSC TRUE/FALSE (typetags 'T' / 'F'), used by /chatbox/typing
     *              and bool avatar parameters
     *   string  -> OSC string (typetag 's'), used by string avatar parameters
     *
     * When holdMs > 0 the value is treated as a press and a matching RELEASE is
     * sent afterwards. The release value must match the OSC type of the press,
     * otherwise VRChat ignores it and the input latches forever:
     *
     *   boolean press -> release false  (an int 0 would not clear a bool param)
     *   integer press -> release 0      (int32, correct for /input/<Button>)
     *   float   press -> release {type:'float', value:0} so the typetag stays
     *                    ',f'; a bare 0 would be downgraded to int32 by
     *                    node-osc and silently ignored on an axis address
     *                    (this is the same P1 axis bug holdAxis() guards).
     *   string  press -> hold is NOT applicable (there is no meaningful "unset"
     *                    string); holdMs is ignored and the caller is told so.
     *
     * @param address - Full OSC address, e.g. '/input/Jump' (CASE SENSITIVE)
     * @param value - Value to send: number, boolean or string
     * @param holdMs - Optional hold time in ms before sending the release value
     * @param ctx - MCP Context (optional)
     * @returns Promise resolving to a confirmation message
     */
    async sendRawOsc(address, value, holdMs = 0, ctx) {
        if (!address || typeof address !== 'string' || address.charAt(0) !== '/') {
            return \`Invalid OSC address: \${address} (must start with '/')\`;
        }
        const valueType = typeof value;
        if (valueType !== 'number' && valueType !== 'boolean' && valueType !== 'string') {
            return \`Invalid OSC value for \${address}: expected number, boolean or string, got \${valueType}\`;
        }
        if (valueType === 'number' && !Number.isFinite(value)) {
            return \`Invalid OSC value for \${address}: \${value} is not a finite number\`;
        }
        if (ctx) {
            await ctx.info(\`Sending raw OSC \${address} = \${JSON.stringify(value)} (\${valueType})\`);
        }
        try {
            const success = await this.wsClient.send_message(address, value);
            if (!success) {
                return \`Failed to send raw OSC \${address} = \${JSON.stringify(value)}\`;
            }
            const hold = Number(holdMs);
            const wantsHold = Number.isFinite(hold) && hold > 0;
            if (wantsHold && valueType === 'string') {
                // No sensible release value exists for a string parameter.
                return \`Sent \${address} = \${JSON.stringify(value)} (string); hold_ms ignored - hold is not applicable to string values\`;
            }
            if (wantsHold) {
                // Release value must carry the SAME OSC type as the press.
                let releaseValue;
                if (valueType === 'boolean') {
                    releaseValue = false;
                }
                else if (Number.isInteger(value)) {
                    releaseValue = 0;
                }
                else {
                    releaseValue = { type: 'float', value: 0 };
                }
                await delay(hold);
                const released = await this.wsClient.send_message(address, releaseValue);
                if (!released) {
                    return \`Sent \${address} = \${JSON.stringify(value)} but failed to release after \${hold}ms\`;
                }
                const releaseLabel = valueType === 'boolean' ? 'false' : '0';
                return \`Sent \${address} = \${JSON.stringify(value)}, released to \${releaseLabel} after \${hold}ms\`;
            }
            return \`Sent \${address} = \${JSON.stringify(value)}\`;
        }
        catch (error) {
            const errorMsg = \`Error sending raw OSC \${address}: \${error instanceof Error ? error.message : String(error)}\`;
            logger.error(errorMsg);
            return errorMsg;
        }
    }
`;
}

/**
 * Build the server.js MCP tool registrations for the BUTTONS table.
 * @returns {string} JavaScript source to splice before the anchor
 */
function buildButtonTools() {
  return BUTTONS.map(function (b) {
    return `server.tool('${b.tool}', '${b.desc}', {
    duration: z.number().default(${b.def}).describe('Duration in seconds to hold the input')
}, async ({ duration }, extra) => {
    try {
        const ctx = createToolContext(extra);
        const result = await inputTools.${b.method}(duration, ctx);
        return { content: [{ type: 'text', text: result }] };
    }
    catch (error) {
        return {
            content: [{
                    type: 'text',
                    text: \`Error in ${b.tool}: \${error instanceof Error ? error.message : String(error)}\`
                }],
            isError: true
        };
    }
});
`;
  }).join('');
}

/**
 * Build the server.js MCP tool registrations for axis / gesture / typing / raw OSC.
 * @returns {string} JavaScript source to splice before the anchor
 */
function buildExtraTools() {
  return `server.tool('move_axis', 'Analog movement: drive /input/Vertical (forward/back) or /input/Horizontal (strafe) with a value in [-1,1], then auto-reset to 0. Use a small value to walk slowly. Set run=true to sprint (holds the Run modifier for the whole movement).', {
    axis: z.enum(['vertical', 'horizontal']).describe('vertical = forward/back, horizontal = strafe right/left'),
    value: z.number().default(1.0).describe('Axis value in [-1.0, 1.0]; positive = forward / right'),
    duration: z.number().default(0.5).describe('Seconds to hold before resetting the axis to 0'),
    run: z.boolean().default(false).describe('Hold the Run modifier while moving (sprint). Run alone does nothing in VRChat - it must overlap with movement.')
}, async ({ axis, value, duration, run }, extra) => {
    try {
        const ctx = createToolContext(extra);
        const result = await inputTools.setMoveAxis(axis, value, duration, run, ctx);
        return { content: [{ type: 'text', text: result }] };
    }
    catch (error) {
        return {
            content: [{
                    type: 'text',
                    text: \`Error in move_axis: \${error instanceof Error ? error.message : String(error)}\`
                }],
            isError: true
        };
    }
});
server.tool('look_axis', 'Analog turning: drive /input/LookHorizontal (yaw) or /input/LookVertical (pitch) with a value in [-1,1], then auto-reset to 0. Finer control than look_direction.', {
    axis: z.enum(['horizontal', 'vertical']).describe('horizontal = turn left/right, vertical = tilt up/down'),
    value: z.number().default(1.0).describe('Axis value in [-1.0, 1.0]; positive = right / up'),
    duration: z.number().default(0.3).describe('Seconds to hold before resetting the axis to 0')
}, async ({ axis, value, duration }, extra) => {
    try {
        const ctx = createToolContext(extra);
        const result = await inputTools.setLookAxis(axis, value, duration, ctx);
        return { content: [{ type: 'text', text: result }] };
    }
    catch (error) {
        return {
            content: [{
                    type: 'text',
                    text: \`Error in look_axis: \${error instanceof Error ? error.message : String(error)}\`
                }],
            isError: true
        };
    }
});
server.tool('set_gesture', 'Set a standard VRChat hand gesture (GestureLeft / GestureRight, Int 0-7: 0 Neutral, 1 Fist, 2 HandOpen, 3 FingerPoint, 4 Victory, 5 RockNRoll, 6 HandGun, 7 ThumbsUp). Persists until set back to 0.', {
    side: z.enum(['left', 'right']).describe('Which hand to set'),
    gesture: z.number().int().min(0).max(7).describe('Gesture index 0-7')
}, async ({ side, gesture }, extra) => {
    try {
        const ctx = createToolContext(extra);
        const result = await inputTools.setGesture(side, gesture, ctx);
        return { content: [{ type: 'text', text: result }] };
    }
    catch (error) {
        return {
            content: [{
                    type: 'text',
                    text: \`Error in set_gesture: \${error instanceof Error ? error.message : String(error)}\`
                }],
            isError: true
        };
    }
});
server.tool('set_typing_indicator', 'Show or hide the VRChat chatbox typing indicator (/chatbox/typing).', {
    typing: z.boolean().describe('true = show the typing indicator, false = hide it')
}, async ({ typing }, extra) => {
    try {
        const ctx = createToolContext(extra);
        const result = await inputTools.setTypingIndicator(typing, ctx);
        return { content: [{ type: 'text', text: result }] };
    }
    catch (error) {
        return {
            content: [{
                    type: 'text',
                    text: \`Error in set_typing_indicator: \${error instanceof Error ? error.message : String(error)}\`
                }],
            isError: true
        };
    }
});
server.tool('send_raw_osc', 'Send a raw OSC message to any VRChat address (escape hatch for inputs/parameters without a dedicated tool). Addresses are CASE SENSITIVE, e.g. /input/Jump or /avatar/parameters/MyParam. Value may be a number (int32 if whole, float32 if fractional), a boolean (OSC TRUE/FALSE) or a string. With hold_ms > 0 the value is sent as a press and a type-matched release follows: boolean releases to false, number releases to 0 (a fractional press releases as float32 so axes actually stop). Strings have no release semantics, so hold_ms is ignored for them.', {
    address: z.string().describe("Full OSC address starting with '/', e.g. /input/Run"),
    value: z.union([z.number(), z.boolean(), z.string()]).describe('Value to send: number for axes/buttons, boolean for toggles (e.g. /chatbox/typing), string for text params (e.g. /avatar/parameters/MyStringParam)'),
    hold_ms: z.number().default(0).describe('Optional hold time in ms before sending the type-matched release value (boolean -> false, number -> 0). Ignored for string values, which have no release semantics.')
}, async ({ address, value, hold_ms }, extra) => {
    try {
        const ctx = createToolContext(extra);
        const result = await inputTools.sendRawOsc(address, value, hold_ms, ctx);
        return { content: [{ type: 'text', text: result }] };
    }
    catch (error) {
        return {
            content: [{
                    type: 'text',
                    text: \`Error in send_raw_osc: \${error instanceof Error ? error.message : String(error)}\`
                }],
            isError: true
        };
    }
});
`;
}

function main() {
  if (!fs.existsSync(SERVER_PATH)) {
    console.warn('[patch-krekun-osc-expand] package server.js not found, skip');
    process.exit(0);
  }

  // ---- 1) inputTools.js: add button + axis + gesture + typing + raw methods ----
  const toolsSrc = fs.readFileSync(INPUT_TOOLS_PATH, 'utf8');
  if (toolsSrc.indexOf(MARKER_TOOLS) !== -1) {
    console.log('[patch-krekun-osc-expand] inputTools.js already patched, skip');
  } else {
    // Hard dependency: pressHoldRelease() comes from patch-krekun-osc-crouch.js.
    if (toolsSrc.indexOf('async pressHoldRelease(') === -1) {
      throw new Error(
        '[patch-krekun-osc-expand] pressHoldRelease() missing in inputTools.js - ' +
        'run scripts/patch-krekun-osc-crouch.js first (check the postinstall order ' +
        'in node_bridge/package.json)'
      );
    }
    // Anchor: the JSDoc block before sendChatboxMessage(), the last method of the
    // class. Inserting before it keeps the new methods inside the class body.
    const anchor = '    /**\n     * Send a message to the VRChat chatbox.';
    const methods = '    ' + MARKER_TOOLS + '\n' + buildButtonMethods() + buildExtraMethods();
    insertBefore(INPUT_TOOLS_PATH, anchor, methods);
    console.log(
      '[patch-krekun-osc-expand] inputTools.js: added ' + BUTTONS.length +
      ' button methods + holdAxis/setMoveAxis/setLookAxis/setGesture/' +
      'setTypingIndicator/sendRawOsc'
    );
  }

  // ---- 2) server.js: register the matching MCP tools ----
  const serverSrc = fs.readFileSync(SERVER_PATH, 'utf8');
  if (serverSrc.indexOf(MARKER_SERVER) !== -1) {
    console.log('[patch-krekun-osc-expand] server.js already patched, skip');
  } else {
    // Anchor: the upstream send_message tool registration (the last one in the
    // file). Inserting before it keeps registration order stable.
    const anchor = "server.tool('send_message', 'Send a message to the VRChat chatbox.', {";
    const tools = MARKER_SERVER + '\n' + buildButtonTools() + buildExtraTools();
    insertBefore(SERVER_PATH, anchor, tools);
    console.log(
      '[patch-krekun-osc-expand] server.js: registered ' + (BUTTONS.length + 5) +
      ' new MCP tools (buttons + move_axis/look_axis/set_gesture/' +
      'set_typing_indicator/send_raw_osc)'
    );
  }

  console.log('[patch-krekun-osc-expand] all patches applied (idempotent markers written)');
}

main();
