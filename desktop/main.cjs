const {app, BrowserWindow, session} = require('electron');
const origin = process.env.SOULX_UI_URL;
let window;
let stopping = false;

app.whenReady().then(() => {
    const trusted = value => {
        try { return new URL(value).origin === origin; } catch { return false; }
    };
    session.defaultSession.setPermissionRequestHandler((contents, permission, callback) => {
        callback(trusted(contents.getURL()) && permission === 'media');
    });
    session.defaultSession.setPermissionCheckHandler((contents, permission, requestingOrigin) => {
        return trusted(requestingOrigin) && permission === 'media';
    });
    window = new BrowserWindow({
        width: 1180, height: 800, minWidth: 860, minHeight: 620,
        title: 'SoulX Agent', backgroundColor: '#101722',
        webPreferences: {contextIsolation: true, nodeIntegration: false, sandbox: true}
    });
    window.webContents.setWindowOpenHandler(() => ({action: 'deny'}));
    window.webContents.on('will-navigate', (event, url) => {
        if (!trusted(url)) event.preventDefault();
    });
    const load = () => {
        if (stopping) return;
        window.loadURL(`${origin}/#${process.env.SOULX_DESKTOP_KEY}`).catch(() => {
            if (!stopping) setTimeout(load, 1000);
        });
    };
    load();
    window.on('closed', () => { stopping = true; app.quit(); });
});
app.on('window-all-closed', () => app.quit());
