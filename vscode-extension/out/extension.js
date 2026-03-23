"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
/**
 * extension.ts — VS Code extension entry point.
 *
 * This file is the only TypeScript file in the extension. It is responsible
 * for starting the Python language server as a child process and connecting
 * it to VS Code's language client infrastructure via the
 * vscode-languageclient library.
 *
 * Architecture
 * ─────────────
 *   VS Code editor
 *       │  vscode-languageclient (npm)
 *       │  JSON-RPC over stdio
 *       ▼
 *   Python child process  ←  server/main.py --mode lsp
 *       │
 *       ├─ GHCBridge (server/ghc/bridge.py)
 *       └─ AIFeedbackEngine (server/ai/engine.py)
 *
 * The extension itself is thin — it does three things:
 *   1. Resolves the Python interpreter and server script path
 *   2. Starts the language server process
 *   3. Registers a "Restart" command for when the server hangs
 *
 * Configuration is forwarded from VS Code settings to the server
 * via workspace/didChangeConfiguration notifications, which pygls
 * handles automatically.
 */
const path = require("path");
const fs = require("fs");
const vscode_1 = require("vscode");
const node_1 = require("vscode-languageclient/node");
let client;
let outputChannel;
async function activate(context) {
    outputChannel = vscode_1.window.createOutputChannel('Haskell AI LSP');
    outputChannel.appendLine('=== Haskell AI Language Server activating ===');
    outputChannel.show(true);
    const config = vscode_1.workspace.getConfiguration('haskellAiLsp');
    if (!config.get('enable', true)) {
        outputChannel.appendLine('Extension disabled via settings.');
        return;
    }
    const pythonPath = resolvePythonPath(outputChannel);
    const serverMain = resolveServerMain(context, outputChannel);
    outputChannel.appendLine(`Python path: ${pythonPath}`);
    outputChannel.appendLine(`Server main: ${serverMain}`);
    outputChannel.appendLine(`Server main exists: ${fs.existsSync(serverMain)}`);
    if (!fs.existsSync(serverMain)) {
        const msg = `Cannot find server/main.py at: ${serverMain}\n\nPlease set haskellAiLsp.projectPath in VS Code settings to the full path of your ai-haskell-lsp folder.`;
        outputChannel.appendLine('ERROR: ' + msg);
        vscode_1.window.showErrorMessage(msg);
        // Still register commands so they don't show "not found"
        registerCommands(context, outputChannel);
        return;
    }
    try {
        client = await createClient(pythonPath, serverMain, outputChannel);
        await client.start();
        outputChannel.appendLine('Language server started successfully.');
    }
    catch (err) {
        outputChannel.appendLine(`ERROR starting server: ${err}`);
        vscode_1.window.showErrorMessage(`Haskell AI LSP failed to start: ${err}. Check the "Haskell AI LSP" output channel.`);
    }
    registerCommands(context, outputChannel);
}
function registerCommands(context, channel) {
    context.subscriptions.push(vscode_1.commands.registerCommand('haskell-ai-lsp.restart', async () => {
        channel.appendLine('Restarting language server…');
        try {
            await client?.stop();
        }
        catch { }
        const pythonPath = resolvePythonPath(channel);
        const serverMain = resolveServerMain(context, channel);
        if (fs.existsSync(serverMain)) {
            client = await createClient(pythonPath, serverMain, channel);
            await client.start();
            channel.appendLine('Language server restarted.');
        }
        else {
            channel.appendLine('ERROR: Cannot find server/main.py — check projectPath setting.');
        }
    }));
    context.subscriptions.push(vscode_1.commands.registerCommand('haskell-ai-lsp.showExplanation', (_uri, _line, explanation, hint) => {
        const msg = [explanation, hint ? `Hint: ${hint}` : '']
            .filter(Boolean)
            .join('\n\n');
        vscode_1.window.showInformationMessage(msg, { modal: false });
    }));
}
function deactivate() {
    return client?.stop();
}
async function createClient(pythonPath, serverMain, channel) {
    const serverOptions = {
        run: {
            command: pythonPath,
            args: [serverMain, '--mode', 'lsp'],
            transport: node_1.TransportKind.stdio,
        },
        debug: {
            command: pythonPath,
            args: [serverMain, '--mode', 'lsp', '--log-level', 'DEBUG'],
            transport: node_1.TransportKind.stdio,
        },
    };
    const clientOptions = {
        documentSelector: [
            { scheme: 'file', language: 'haskell' },
            { scheme: 'untitled', language: 'haskell' },
        ],
        synchronize: {
            fileEvents: vscode_1.workspace.createFileSystemWatcher('**/*.hs'),
            configurationSection: 'haskellAiLsp',
        },
        outputChannel: channel,
        revealOutputChannelOn: node_1.RevealOutputChannelOn.Error,
        initializationOptions: getInitOptions(),
    };
    return new node_1.LanguageClient('haskell-ai-lsp', 'Haskell AI Language Server', serverOptions, clientOptions);
}
function resolvePythonPath(channel) {
    const config = vscode_1.workspace.getConfiguration('haskellAiLsp');
    // 1. Explicit setting
    const explicit = config.get('serverPath', '').trim();
    if (explicit) {
        channel.appendLine(`Using explicit Python path: ${explicit}`);
        return explicit;
    }
    // 2. Python extension interpreter
    try {
        const pythonConfig = vscode_1.workspace.getConfiguration('python');
        const interpreter = pythonConfig.get('defaultInterpreterPath', '').trim();
        if (interpreter && fs.existsSync(interpreter)) {
            channel.appendLine(`Using Python extension interpreter: ${interpreter}`);
            return interpreter;
        }
    }
    catch { }
    // 3. Windows: try 'python' before 'python3' (python3 often not found on Windows)
    const isWindows = process.platform === 'win32';
    const fallback = isWindows ? 'python' : 'python3';
    channel.appendLine(`Using fallback Python: ${fallback}`);
    return fallback;
}
function resolveServerMain(context, channel) {
    const config = vscode_1.workspace.getConfiguration('haskellAiLsp');
    // 1. Explicit projectPath setting
    const projectPath = config.get('projectPath', '').trim();
    if (projectPath) {
        const candidate = path.join(projectPath, 'server', 'main.py');
        channel.appendLine(`Trying projectPath setting: ${candidate}`);
        if (fs.existsSync(candidate)) {
            return candidate;
        }
        channel.appendLine(`WARNING: projectPath set but server/main.py not found there`);
    }
    // 2. Workspace root
    const folders = vscode_1.workspace.workspaceFolders;
    if (folders && folders.length > 0) {
        const candidate = path.join(folders[0].uri.fsPath, 'server', 'main.py');
        channel.appendLine(`Trying workspace root: ${candidate}`);
        if (fs.existsSync(candidate)) {
            return candidate;
        }
    }
    // 3. Bundled inside extension (for distributed .vsix)
    const bundled = path.join(context.extensionPath, 'server', 'main.py');
    channel.appendLine(`Trying bundled path: ${bundled}`);
    return bundled;
}
function getInitOptions() {
    const config = vscode_1.workspace.getConfiguration('haskellAiLsp');
    return {
        ghcPath: config.get('ghcPath', ''),
        groqModel: config.get('groqModel', 'llama-3.1-8b-instant'),
    };
}
//# sourceMappingURL=extension.js.map