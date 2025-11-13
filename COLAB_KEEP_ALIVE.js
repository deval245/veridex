// ============================================================================
// COLAB KEEP-ALIVE SCRIPT
// Prevents Colab from disconnecting due to "idle" detection
// ============================================================================

// PASTE THIS INTO YOUR BROWSER'S JAVASCRIPT CONSOLE WHILE COLAB IS RUNNING
// (Press F12 → Console tab → Paste this code → Press Enter)

function KeepAlive() {
    console.log("🟢 Keep-Alive: Simulating activity...");
    
    // Click on the output area to prevent idle detection
    const outputArea = document.querySelector('colab-output-area');
    if (outputArea) {
        outputArea.click();
    }
    
    // Move mouse cursor programmatically
    document.dispatchEvent(new MouseEvent('mousemove', {
        view: window,
        bubbles: true,
        cancelable: true
    }));
    
    // Press Ctrl+M I (harmless command that shows keyboard shortcuts)
    document.dispatchEvent(new KeyboardEvent('keydown', {
        key: 'm',
        ctrlKey: true,
        bubbles: true
    }));
}

// Run every 60 seconds
const keepAliveInterval = setInterval(KeepAlive, 60000);

console.log("✅ Keep-Alive activated!");
console.log("⏱️  Simulating activity every 60 seconds");
console.log("🛑 To stop: Run 'clearInterval(keepAliveInterval)' in console");

// Also prevent browser tab from sleeping
const preventSleep = () => {
    fetch('/api/sessions')
        .then(() => console.log('🔄 Session pinged'))
        .catch(() => console.log('⚠️ Session ping failed'));
};

const sessionPingInterval = setInterval(preventSleep, 300000); // Every 5 minutes

console.log("✅ Session pinger activated!");










