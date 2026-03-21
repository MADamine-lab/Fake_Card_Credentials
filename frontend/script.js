// ─── API Config ────────────────────────────────────────────────────────────
const API_BASE_URL = 'http://localhost:5000/api';

// ─── State ─────────────────────────────────────────────────────────────────
let currentMessageId = null;
let selectedImage    = null;
let scanCount        = 0;
let fakeCount        = 0;
let realCount        = 0;
let clearCount       = 0;

// ─── DOM References ────────────────────────────────────────────────────────
const chatContainer  = document.getElementById('chatContainer');
const sendButton     = document.getElementById('sendButton');
const browseButton   = document.getElementById('browseButton');
const imageInput     = document.getElementById('imageInput');
const imagePreview   = document.getElementById('imagePreview');
const previewImage   = document.getElementById('previewImage');
const removeImageBtn = document.getElementById('removeImage');
const dropZone       = document.getElementById('dropZone');
const authModal      = document.getElementById('authModal');
const loadingOverlay = document.getElementById('loadingOverlay');
const statusDot      = document.getElementById('statusDot');
const statusText     = document.getElementById('statusText');
const modelDot       = document.getElementById('modelDot');
const modelText      = document.getElementById('modelText');

// ─── Init ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    checkServerStatus();
    setupEventListeners();
    setupDragAndDrop();
});

// ─── Event Listeners ───────────────────────────────────────────────────────
function setupEventListeners() {
    sendButton.addEventListener('click', handleSendImage);
    browseButton.addEventListener('click', () => imageInput.click());
    imageInput.addEventListener('change', handleImageSelect);
    removeImageBtn.addEventListener('click', removeImage);

    document.getElementById('cancelAuth').addEventListener('click', closeAuthModal);
    document.getElementById('verifyAuth').addEventListener('click', handleVerifyCode);

    document.getElementById('authCodeInput').addEventListener('input', (e) => {
        e.target.value = e.target.value.replace(/\D/g, '');
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeAuthModal();
    });
}

// ─── Drag & Drop ───────────────────────────────────────────────────────────
function setupDragAndDrop() {
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) loadImageFile(file);
        else showToast('Veuillez déposer un fichier image.', 'error');
    });
}

// ─── Server / Model status ─────────────────────────────────────────────────
async function checkServerStatus() {
    try {
        const res  = await fetch(`${API_BASE_URL}/health`);
        const data = await res.json();
        const online      = data.status === 'healthy';
        const modelLoaded = data.models?.fake_cred_detector ?? false;
        updateStatus(online);
        updateModelStatus(modelLoaded);
    } catch {
        updateStatus(false);
        updateModelStatus(false);
    }
}

function updateStatus(online) {
    statusDot.classList.toggle('offline', !online);
    statusText.textContent = online ? 'Connecté' : 'Hors ligne';
}

function updateModelStatus(loaded) {
    modelDot.classList.toggle('model-loaded', loaded);
    modelDot.classList.toggle('model-error', !loaded);
    modelText.textContent = loaded ? 'Modèle YOLOv5 prêt' : 'Modèle non chargé';
}

// ─── Stat counters ─────────────────────────────────────────────────────────
function updateStats() {
    document.getElementById('scanCount').textContent  = scanCount;
    document.getElementById('fakeCount').textContent  = fakeCount;
    document.getElementById('realCount').textContent  = realCount;
    document.getElementById('clearCount').textContent = clearCount;
}

function updateThreatLevel() {
    const total = scanCount || 1;
    const ratio = fakeCount / total;
    const pct   = Math.round(ratio * 100);

    const fill  = document.getElementById('threatFill');
    const label = document.getElementById('threatValue');

    fill.style.width = `${Math.max(5, pct)}%`;

    if (ratio < 0.15) {
        fill.style.background = 'linear-gradient(90deg,#00e096,#00d4ff)';
        label.style.color     = 'var(--accent-green)';
        label.textContent     = 'LOW';
    } else if (ratio < 0.4) {
        fill.style.background = 'linear-gradient(90deg,#ffb547,#ff8c42)';
        label.style.color     = 'var(--accent-warn)';
        label.textContent     = 'MEDIUM';
    } else {
        fill.style.background = 'linear-gradient(90deg,#ff4d6d,#c0392b)';
        label.style.color     = 'var(--accent-danger)';
        label.textContent     = 'HIGH';
    }
}

// ─── Image handling ────────────────────────────────────────────────────────
function handleImageSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
        showToast('Veuillez sélectionner un fichier image.', 'error');
        return;
    }
    loadImageFile(file);
}

function loadImageFile(file) {
    const reader = new FileReader();
    reader.onload = (ev) => {
        selectedImage              = ev.target.result;
        previewImage.src           = selectedImage;
        imagePreview.style.display = 'flex';
        dropZone.style.display     = 'none';
        sendButton.disabled        = false;
    };
    reader.readAsDataURL(file);
}

function removeImage() {
    selectedImage              = null;
    imagePreview.style.display = 'none';
    dropZone.style.display     = 'flex';
    previewImage.src           = '';
    imageInput.value           = '';
    sendButton.disabled        = true;
}

// ─── Send image for analysis ───────────────────────────────────────────────
async function handleSendImage() {
    if (!selectedImage) return;

    const tempImage = selectedImage;
    removeImage();

    // Show submitted image card
    addImageCard(tempImage);
    scanCount++;
    updateStats();
    updateThreatLevel();

    showLoading(true);

    try {
        const res  = await fetch(`${API_BASE_URL}/send-message`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ image: tempImage }),
            credentials: 'include'
        });
        const data = await res.json();

        showLoading(false);

        if (data.status === 'success') {
            clearCount++;
            addResultCard('clear', 'Aucune carte détectée', 'L\'image ne contient pas de carte bancaire.');
        } else if (data.status === 'pending') {
            // Count by issue types
            const issues = data.detected_issues || [];
            const hasFake = issues.some(i => i.type === 'CARTE_FALSIFIEE');
            const hasReal = issues.some(i => i.type === 'CARTE_REELLE');
            if (hasFake) fakeCount++;
            else if (hasReal) realCount++;

            currentMessageId = data.message_id;
            addResultCard('flagged', 'Carte détectée — Authentification requise', buildIssuesSummary(issues));
            showAuthModal(data);
        } else {
            addResultCard('error', 'Erreur', data.error || "Erreur inconnue lors de l'analyse.");
        }

        updateStats();
        updateThreatLevel();

    } catch (err) {
        console.error('[SendImage]', err);
        showLoading(false);
        addResultCard('error', 'Erreur de connexion', 'Impossible de joindre le serveur d\'analyse.');
    }
}

function buildIssuesSummary(issues) {
    if (!issues || issues.length === 0) return 'Contenu sensible détecté.';
    return issues.map(i => {
        let s = i.message || i.type;
        if (i.confidence) s += ` (${(i.confidence * 100).toFixed(1)}% confiance)`;
        return s;
    }).join(' · ');
}

// ─── Add cards to the feed ─────────────────────────────────────────────────
function removeWelcome() {
    const w = document.getElementById('welcomeScreen');
    if (w) w.remove();
}

function addImageCard(imageSrc) {
    removeWelcome();
    const wrap = document.createElement('div');
    wrap.className = 'result-card submitted';
    wrap.innerHTML = `
        <div class="result-card-head">
            <span class="result-tag submitted">SOUMIS</span>
            <span class="result-time">${now()}</span>
        </div>
        <img src="${imageSrc}" class="result-thumb" alt="Image soumise">
    `;
    chatContainer.appendChild(wrap);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addResultCard(type, title, detail) {
    const wrap = document.createElement('div');
    wrap.className = `result-card ${type}`;

    const iconMap = {
        clear:   `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`,
        flagged: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3z"/><line x1="12" y1="9" x2="12" y2="13"/></svg>`,
        error:   `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/></svg>`,
    };

    const tagLabel = { clear: 'SÉCURISÉ', flagged: 'ALERTE', error: 'ERREUR' };

    wrap.innerHTML = `
        <div class="result-card-head">
            <div class="result-tag-wrap">
                <span class="result-tag ${type}">${tagLabel[type] || type.toUpperCase()}</span>
                <span class="result-icon ${type}">${iconMap[type] || ''}</span>
            </div>
            <span class="result-time">${now()}</span>
        </div>
        <div class="result-title">${title}</div>
        <div class="result-detail">${detail}</div>
    `;
    chatContainer.appendChild(wrap);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function now() {
    return new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// ─── Auth Modal ────────────────────────────────────────────────────────────
function showAuthModal(data) {
    const issuesPanel   = document.getElementById('detectedIssues');
    const emailNotif    = document.getElementById('emailNotification');
    const codeContainer = document.getElementById('codeDisplayContainer');
    const codeDisplay   = document.getElementById('authCodeDisplay');

    issuesPanel.innerHTML = '';
    emailNotif.innerHTML  = '';
    emailNotif.style.display = 'none';

    if (data.detected_issues && data.detected_issues.length > 0) {
        data.detected_issues.forEach(issue => {
            const item = document.createElement('div');
            item.className = 'issue-item';

            const typeClass = issue.type === 'CARTE_FALSIFIEE' ? 'danger' : '';
            item.innerHTML = `
                <span class="issue-type ${typeClass}">${issue.type || 'ALERTE'}</span>
                <div>
                    <div>${issue.message || ''}</div>
                    ${issue.confidence
                        ? `<div class="issue-conf">Confiance : ${(issue.confidence * 100).toFixed(1)} %</div>`
                        : ''}
                    ${issue.count
                        ? `<div class="issue-conf">${issue.count} détectée(s)</div>`
                        : ''}
                </div>`;
            issuesPanel.appendChild(item);
        });
    }

    if (data.email_sent) {
        codeContainer.style.display = 'none';
        emailNotif.style.display    = 'block';
        emailNotif.className        = 'notif-box success';
        emailNotif.innerHTML        = `
            <strong>Email envoyé</strong><br>
            ${data.email_message || 'Un code a été envoyé à votre adresse email.'}<br>
            <small style="opacity:.7">Vérifiez aussi vos spams.</small>`;
    } else if (data.auth_code) {
        codeContainer.style.display = 'block';
        codeDisplay.textContent     = data.auth_code;

        if (data.email_sent === false && data.email_message) {
            emailNotif.style.display = 'block';
            emailNotif.className     = 'notif-box warning';
            emailNotif.innerHTML     = `
                <strong>Échec de l'envoi email</strong><br>
                ${data.email_message}<br>
                <small style="opacity:.7">Code affiché en mode développement.</small>`;
        }
    } else {
        codeContainer.style.display = 'none';
        showToast('Erreur : code non disponible', 'error');
    }

    document.getElementById('authCodeInput').value = '';
    authModal.classList.add('active');
    setTimeout(() => document.getElementById('authCodeInput').focus(), 200);
}

function closeAuthModal() {
    authModal.classList.remove('active');
    currentMessageId = null;
    const codeContainer = document.getElementById('codeDisplayContainer');
    const emailNotif    = document.getElementById('emailNotification');
    if (codeContainer) codeContainer.style.display = 'block';
    if (emailNotif)    { emailNotif.style.display = 'none'; emailNotif.innerHTML = ''; }
}

// ─── Verify OTP ────────────────────────────────────────────────────────────
async function handleVerifyCode() {
    const code = document.getElementById('authCodeInput').value.trim();
    if (!code || code.length !== 6) {
        showToast('Veuillez entrer un code à 6 chiffres.', 'error');
        return;
    }

    showLoading(true);

    try {
        const res  = await fetch(`${API_BASE_URL}/verify-code`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ message_id: currentMessageId, code }),
            credentials: 'include'
        });
        const data = await res.json();

        showLoading(false);

        if (data.status === 'success') {
            addResultCard('clear', 'Accès autorisé', 'Identité vérifiée — image analysée et consignée.');
            closeAuthModal();
            updateStats();
        } else {
            showToast(data.error || 'Code incorrect. Veuillez réessayer.', 'error');
        }
    } catch (err) {
        console.error('[VerifyCode]', err);
        showLoading(false);
        showToast('Erreur de vérification.', 'error');
    }
}

// ─── Loading ───────────────────────────────────────────────────────────────
function showLoading(show) {
    loadingOverlay.classList.toggle('active', show);
}

// ─── Toast ─────────────────────────────────────────────────────────────────
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<div class="toast-dot"></div><span>${message}</span>`;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'toastOut 0.3s ease forwards';
        setTimeout(() => toast.remove(), 320);
    }, 3200);
}

// ─── Periodic status check ─────────────────────────────────────────────────
setInterval(checkServerStatus, 30_000);