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

import * as path from 'path';
import * as fs from 'fs';
import {
    workspace,
    ExtensionContext,
    window,
    commands,
    OutputChannel,
} from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind,
    RevealOutputChannelOn,
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;
let outputChannel: OutputChannel | undefined;

// ── Activation ────────────────────────────────────────────────────────────

export async function activate(context: ExtensionContext): Promise<void> {
    outputChannel = window.createOutputChannel('Haskell AI LSP');
    outputChannel.appendLine('Haskell AI Language Server starting…');

    const config = workspace.getConfiguration('haskellAiLsp');
    if (!config.get<boolean>('enable', true)) {
        outputChannel.appendLine('Extension disabled via settings.');
        return;
    }

    try {
        client = await createClient(context, outputChannel);
        await client.start();
        outputChannel.appendLine('Language server started.');
    } catch (err) {
        window.showErrorMessage(
            `Haskell AI LSP failed to start: ${err}. ` +
            'Check the "Haskell AI LSP" output channel for details.'
        );
    }

    // Register commands
    context.subscriptions.push(
        commands.registerCommand('haskell-ai-lsp.restart', async () => {
            outputChannel?.appendLine('Restarting language server…');
            await client?.stop();
            client = await createClient(context, outputChannel!);
            await client.start();
            outputChannel?.appendLine('Language server restarted.');
        })
    );

    context.subscriptions.push(
        commands.registerCommand(
            'haskell-ai-lsp.showExplanation',
            (_uri: string, _line: number, explanation: string, hint: string) => {
                // Show the AI explanation in an information message pop-up.
                // This is triggered by the code action "💡 AI: ..." button.
                const msg = [explanation, hint ? `Hint: ${hint}` : '']
                    .filter(Boolean)
                    .join('\n\n');
                window.showInformationMessage(msg, { modal: false });
            }
        )
    );
}

// ── Deactivation ──────────────────────────────────────────────────────────

export function deactivate(): Thenable<void> | undefined {
    return client?.stop();
}

// ── Client factory ────────────────────────────────────────────────────────

async function createClient(
    context: ExtensionContext,
    channel: OutputChannel,
): Promise<LanguageClient> {

    const pythonPath  = resolvePythonPath();
    const serverMain  = resolveServerMain(context);

    channel.appendLine(`Python: ${pythonPath}`);
    channel.appendLine(`Server: ${serverMain}`);

    // Server options: spawn Python with server/main.py as a child process.
    // Communication is over stdio (stdin/stdout). All logging goes to stderr,
    // which VS Code captures and shows in the output channel.
    const serverOptions: ServerOptions = {
        run: {
            command:   pythonPath,
            args:      [serverMain, '--mode', 'lsp'],
            transport: TransportKind.stdio,
        },
        debug: {
            command:   pythonPath,
            args:      [serverMain, '--mode', 'lsp', '--log-level', 'DEBUG'],
            transport: TransportKind.stdio,
        },
    };

    // Client options: activate for Haskell files only.
    const clientOptions: LanguageClientOptions = {
        documentSelector: [
            { scheme: 'file', language: 'haskell' },
            { scheme: 'untitled', language: 'haskell' },
        ],
        synchronize: {
            // Watch .hs files for changes so the server can respond to
            // file saves outside the active editor
            fileEvents: workspace.createFileSystemWatcher('**/*.hs'),
            // Forward configuration changes to the server automatically
            configurationSection: 'haskellAiLsp',
        },
        outputChannel: channel,
        revealOutputChannelOn: RevealOutputChannelOn.Error,
        initializationOptions: getInitOptions(),
    };

    return new LanguageClient(
        'haskell-ai-lsp',
        'Haskell AI Language Server',
        serverOptions,
        clientOptions,
    );
}

// ── Path resolution ────────────────────────────────────────────────────────

/**
 * Resolve the Python interpreter to use.
 *
 * Priority:
 *   1. haskellAiLsp.serverPath setting
 *   2. The Python interpreter selected in the Python extension (ms-python)
 *   3. 'python3' (system default)
 */
function resolvePythonPath(): string {
    const config = workspace.getConfiguration('haskellAiLsp');
    const explicit = config.get<string>('serverPath', '').trim();
    if (explicit) {
        return explicit;
    }

    // Try to read the active interpreter from the Python extension
    try {
        const pythonConfig = workspace.getConfiguration('python');
        const interpreter  = pythonConfig.get<string>('defaultInterpreterPath', '');
        if (interpreter) {
            return interpreter;
        }
    } catch {
        // Python extension not installed — fall through
    }

    return 'python3';
}

/**
 * Resolve the path to server/main.py.
 *
 * During development the extension is run from the repo root, so we
 * look relative to the workspace folder first, then fall back to the
 * extension's own directory (for packaged .vsix installs where main.py
 * is bundled alongside the extension).
 */
function resolveServerMain(context: ExtensionContext): string {
    // Try workspace root first (development / repo usage)
    const folders = workspace.workspaceFolders;
    if (folders && folders.length > 0) {
        const candidate = path.join(folders[0].uri.fsPath, 'server', 'main.py');
        if (fs.existsSync(candidate)) {
            return candidate;
        }
    }

    // Fall back to bundled path inside the extension directory
    return path.join(context.extensionPath, 'server', 'main.py');
}

/**
 * Build the initialization options forwarded to the Python server on startup.
 * These mirror the workspace configuration so the server has them before the
 * first workspace/didChangeConfiguration notification arrives.
 */
function getInitOptions(): Record<string, unknown> {
    const config = workspace.getConfiguration('haskellAiLsp');
    return {
        ghcPath:    config.get<string>('ghcPath', ''),
        groqModel:  config.get<string>('groqModel', 'llama-3.1-8b-instant'),
    };
}