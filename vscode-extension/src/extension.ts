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

export async function activate(context: ExtensionContext): Promise<void> {
    outputChannel = window.createOutputChannel('Haskell AI LSP');
    outputChannel.appendLine('=== Haskell AI Language Server activating ===');
    outputChannel.show(true);

    const config = workspace.getConfiguration('haskellAiLsp');
    if (!config.get<boolean>('enable', true)) {
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
        window.showErrorMessage(msg);
        // Still register commands so they don't show "not found"
        registerCommands(context, outputChannel);
        return;
    }

    try {
        client = await createClient(pythonPath, serverMain, outputChannel);
        await client.start();
        outputChannel.appendLine('Language server started successfully.');
    } catch (err) {
        outputChannel.appendLine(`ERROR starting server: ${err}`);
        window.showErrorMessage(
            `Haskell AI LSP failed to start: ${err}. Check the "Haskell AI LSP" output channel.`
        );
    }

    registerCommands(context, outputChannel);
}

function registerCommands(context: ExtensionContext, channel: OutputChannel) {
    context.subscriptions.push(
        commands.registerCommand('haskell-ai-lsp.restart', async () => {
            channel.appendLine('Restarting language server…');
            try {
                await client?.stop();
            } catch {}
            const pythonPath = resolvePythonPath(channel);
            const serverMain = resolveServerMain(context, channel);
            if (fs.existsSync(serverMain)) {
                client = await createClient(pythonPath, serverMain, channel);
                await client.start();
                channel.appendLine('Language server restarted.');
            } else {
                channel.appendLine('ERROR: Cannot find server/main.py — check projectPath setting.');
            }
        })
    );

    context.subscriptions.push(
        commands.registerCommand(
            'haskell-ai-lsp.showExplanation',
            (_uri: string, _line: number, explanation: string, hint: string) => {
                const msg = [explanation, hint ? `Hint: ${hint}` : '']
                    .filter(Boolean)
                    .join('\n\n');
                window.showInformationMessage(msg, { modal: false });
            }
        )
    );
}

export function deactivate(): Thenable<void> | undefined {
    return client?.stop();
}

async function createClient(
    pythonPath: string,
    serverMain: string,
    channel: OutputChannel,
): Promise<LanguageClient> {

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

    const clientOptions: LanguageClientOptions = {
        documentSelector: [
            { scheme: 'file',     language: 'haskell' },
            { scheme: 'untitled', language: 'haskell' },
        ],
        synchronize: {
            fileEvents: workspace.createFileSystemWatcher('**/*.hs'),
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

function resolvePythonPath(channel: OutputChannel): string {
    const config = workspace.getConfiguration('haskellAiLsp');

    // 1. Explicit setting
    const explicit = config.get<string>('serverPath', '').trim();
    if (explicit) {
        channel.appendLine(`Using explicit Python path: ${explicit}`);
        return explicit;
    }

    // 2. Python extension interpreter
    try {
        const pythonConfig = workspace.getConfiguration('python');
        const interpreter  = pythonConfig.get<string>('defaultInterpreterPath', '').trim();
        if (interpreter && fs.existsSync(interpreter)) {
            channel.appendLine(`Using Python extension interpreter: ${interpreter}`);
            return interpreter;
        }
    } catch {}

    // 3. Windows: try 'python' before 'python3' (python3 often not found on Windows)
    const isWindows = process.platform === 'win32';
    const fallback  = isWindows ? 'python' : 'python3';
    channel.appendLine(`Using fallback Python: ${fallback}`);
    return fallback;
}

function resolveServerMain(context: ExtensionContext, channel: OutputChannel): string {
    const config = workspace.getConfiguration('haskellAiLsp');

    // 1. Explicit projectPath setting
    const projectPath = config.get<string>('projectPath', '').trim();
    if (projectPath) {
        const candidate = path.join(projectPath, 'server', 'main.py');
        channel.appendLine(`Trying projectPath setting: ${candidate}`);
        if (fs.existsSync(candidate)) {
            return candidate;
        }
        channel.appendLine(`WARNING: projectPath set but server/main.py not found there`);
    }

    // 2. Workspace root
    const folders = workspace.workspaceFolders;
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

function getInitOptions(): Record<string, unknown> {
    const config = workspace.getConfiguration('haskellAiLsp');
    return {
        ghcPath:   config.get<string>('ghcPath', ''),
        groqModel: config.get<string>('groqModel', 'llama-3.1-8b-instant'),
    };
}