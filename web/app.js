const API_BASE = "http://localhost:8000";

// Elements
const preInput = document.getElementById('pre-input');
const postInput = document.getElementById('post-input');
const preDropZone = document.getElementById('pre-drop-zone');
const postDropZone = document.getElementById('post-drop-zone');
const prePreview = document.getElementById('pre-preview');
const postPreview = document.getElementById('post-preview');
const analyzeBtn = document.getElementById('analyze-btn');
const resultsSection = document.getElementById('results-section');
const loader = document.getElementById('loader');
const statusIndicator = document.getElementById('api-status');

let preFile = null;
let postFile = null;

// Check API Health
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/`);
        const data = await res.json();
        if (data.status === "online") {
            statusIndicator.classList.add('online');
            statusIndicator.innerHTML = '<i class="fas fa-circle"></i> AI Engine Ready';
        }
    } catch (e) {
        statusIndicator.classList.add('offline');
        statusIndicator.innerHTML = '<i class="fas fa-circle"></i> Engine Offline';
    }
}

checkHealth();

// Handle Uploads
const setupDropZone = (zone, input, preview, type) => {
    zone.onclick = () => input.click();

    input.onchange = (e) => {
        const file = e.target.files[0];
        if (file) handleFile(file, preview, type);
    };

    zone.ondragover = (e) => { e.preventDefault(); zone.style.background = "rgba(255,255,255,0.05)"; };
    zone.ondragleave = () => { zone.style.background = "transparent"; };
    zone.ondrop = (e) => {
        e.preventDefault();
        zone.style.background = "transparent";
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file, preview, type);
    };
};

function handleFile(file, preview, type) {
    if (type === 'pre') preFile = file;
    else postFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.classList.remove('hidden');
        checkAbility();
    };
    reader.readAsDataURL(file);
}

function checkAbility() {
    analyzeBtn.disabled = !(preFile && postFile);
}

setupDropZone(preDropZone, preInput, prePreview, 'pre');
setupDropZone(postDropZone, postInput, postPreview, 'post');

// Run Analysis
analyzeBtn.onclick = async () => {
    loader.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    const formData = new FormData();
    formData.append('pre_image', preFile);
    formData.append('post_image', postFile);

    try {
        const res = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const errorData = await res.json();
            throw new Error(errorData.detail || "Processing Failed");
        }

        const data = await res.json();
        displayResults(data);
    } catch (e) {
        alert("Assessment Error: " + e.message);
    } finally {
        loader.classList.add('hidden');
    }
};

function displayResults(data) {
    resultsSection.classList.remove('hidden');
    document.getElementById('mask-result').src = data.mask;
    document.getElementById('severity-score').innerText = data.severity;
    document.getElementById('scene-assessment').innerText = data.scene_assessment;
    document.getElementById('confidence-val').innerText = data.confidence;

    const statsContainer = document.getElementById('stats-container');
    statsContainer.innerHTML = '';

    const colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'];
    let i = 0;

    for (const [label, val] of Object.entries(data.distribution)) {
        const div = document.createElement('div');
        div.className = 'stat-item';
        div.innerHTML = `
            <div class="stat-label">
                <span>${label}</span>
                <span>${val}%</span>
            </div>
            <div class="stat-bar">
                <div class="stat-progress" style="width: 0%; background: ${colors[i++]}"></div>
            </div>
        `;
        statsContainer.appendChild(div);

        // Trigger animation
        setTimeout(() => {
            div.querySelector('.stat-progress').style.width = `${val}%`;
        }, 100);
    }

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}
