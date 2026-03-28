const STORAGE_KEY = 'lexigen_history';
let currentResults = null;
let visibleClauses = 5;
let currentActiveId = null;
let pendingDeleteId = null;

// Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('pdf-input');
const fileReadyCard = document.getElementById('file-ready');
const filenameDisplay = document.getElementById('filename-display');
const removeFileBtn = document.getElementById('remove-file');
const startBtn = document.getElementById('start-btn');
const processForm = document.getElementById('process-form');

const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progress-bar');
const progressPercent = document.getElementById('progress-percent');
const statusMessage = document.getElementById('status-message');

const resultsDashboard = document.getElementById('results-dashboard');
const historyList = document.getElementById('history-list');
const newAnalysisBtn = document.getElementById('new-analysis-btn');

const clausesContainer = document.getElementById('clauses-container');
const paginationArea = document.getElementById('pagination-controls');
const loadMoreBtn = document.getElementById('load-more-btn');
const showLessBtn = document.getElementById('show-less-btn');

const tooltip = document.getElementById('custom-tooltip');

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
    setupTooltips();
});

// History Logic
function loadHistory() {
    const history = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    historyList.innerHTML = '';
    
    if (history.length === 0) {
        historyList.innerHTML = '<div class="empty-history">No recent analyses</div>';
        return;
    }

    history.sort((a, b) => b.timestamp - a.timestamp).forEach(item => {
        const div = document.createElement('div');
        div.className = 'history-item';
        if (item.id === currentActiveId) div.classList.add('active');
        
        div.onclick = () => renderResultsFromHistory(item.id);
        
        div.innerHTML = `
            <span class="history-item-label">${item.filename}</span>
            <button class="btn-delete-history" title="Delete results">
                <i class="fas fa-trash-alt"></i>
            </button>
        `;

        // Handle delete specifically
        const deleteBtn = div.querySelector('.btn-delete-history');
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            showDeleteConfirm(item.id, item.filename);
        };

        historyList.appendChild(div);
    });
}

function showDeleteConfirm(id, filename) {
    pendingDeleteId = id;
    document.getElementById('modal-filename').textContent = filename;
    document.getElementById('confirm-modal').classList.remove('hidden');
}

document.getElementById('modal-cancel-btn').onclick = () => {
    document.getElementById('confirm-modal').classList.add('hidden');
    pendingDeleteId = null;
};

document.getElementById('modal-confirm-btn').onclick = () => {
    if (pendingDeleteId) {
        deleteHistoryItem(pendingDeleteId);
        document.getElementById('confirm-modal').classList.add('hidden');
        pendingDeleteId = null;
    }
};

function deleteHistoryItem(id) {
    let history = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    history = history.filter(item => item.id !== id);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    
    if (id === currentActiveId) {
        resetUI();
    }
    loadHistory();
}

function saveToHistory(result, filename) {
    const history = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    const id = Date.now().toString();
    const newItem = {
        id,
        filename,
        timestamp: Date.now(),
        data: result
    };
    history.push(newItem);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    loadHistory();
    return id;
}

function renderResultsFromHistory(id) {
    const history = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
    const item = history.find(h => h.id === id);
    if (item) {
        currentActiveId = id;
        // Clear workspace
        resetUI(false); // Don't clear currentActiveId here
        dropZone.classList.add('hidden');
        renderResults(item.data);
        
        // Re-render history list to update active states
        loadHistory();
    }
}

newAnalysisBtn.onclick = () => {
    resetUI();
};

// Tooltip Logic
function setupTooltips() {
    document.querySelectorAll('.info-icon').forEach(icon => {
        icon.addEventListener('mouseenter', (e) => {
            const msg = icon.getAttribute('data-tooltip');
            tooltip.textContent = msg;
            tooltip.classList.add('active');
            
            const rect = icon.getBoundingClientRect();
            tooltip.style.left = `${rect.left + window.scrollX - (tooltip.offsetWidth / 2)}px`;
            tooltip.style.top = `${rect.top + window.scrollY - tooltip.offsetHeight - 10}px`;
        });
        
        icon.addEventListener('mouseleave', () => {
            tooltip.classList.remove('active');
        });
    });
}

// File Upload Logic
dropZone.onclick = () => fileInput.click();

dropZone.ondragover = (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
};

dropZone.ondragleave = () => dropZone.classList.remove('drag-over');

dropZone.ondrop = (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFileSelect();
    }
};

fileInput.onchange = handleFileSelect;

function handleFileSelect() {
    if (fileInput.files.length) {
        filenameDisplay.textContent = fileInput.files[0].name;
        dropZone.classList.add('hidden');
        fileReadyCard.classList.remove('hidden');
    }
}

removeFileBtn.onclick = () => {
    fileInput.value = '';
    fileReadyCard.classList.add('hidden');
    dropZone.classList.remove('hidden');
};

// Analysis Logic
startBtn.onclick = async () => {
    const formData = new FormData(processForm);
    formData.append('pdf', fileInput.files[0]);

    fileReadyCard.classList.add('hidden');
    progressContainer.classList.remove('hidden');
    startBtn.disabled = true;

    try {
        const response = await fetch('/process', { method: 'POST', body: formData });
        const data = await response.json();
        if (data.task_id) {
            startStatusStream(data.task_id, fileInput.files[0].name);
        } else {
            alert("Error: " + (data.error || "Unknown error"));
            resetUI();
        }
    } catch (err) {
        alert("Failed to connect to server.");
        resetUI();
    }
};

function startStatusStream(taskId, filename) {
    const eventSource = new EventSource(`/status/${taskId}`);

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        statusMessage.textContent = data.message;
        progressBar.style.width = data.progress + "%";
        progressPercent.textContent = data.progress + "%";

        if (data.status === 'completed') {
            eventSource.close();
            saveToHistory(data.result, filename);
            renderResults(data.result);
        } else if (data.status === 'error') {
            eventSource.close();
            alert("Analysis failed: " + data.error);
            resetUI();
        }
    };
}

function renderResults(result) {
    currentResults = result;
    visibleClauses = 5; // Reset pagination
    
    progressContainer.classList.add('hidden');
    resultsDashboard.classList.remove('hidden');
    
    // Summary & Metrics
    document.getElementById('result-summary').textContent = result.summary;
    const eval = result.evaluation || {};
    document.getElementById('bleu-val').textContent = (eval.bleu || 0).toFixed(4);
    document.getElementById('rouge1-val').textContent = (eval.rouge?.rouge1 || 0).toFixed(4);
    document.getElementById('rougel-val').textContent = (eval.rouge?.rougeL || 0).toFixed(4);

    updateClausesDisplay();
    setupTooltips(); // Re-bind tooltips for new elements
}

function updateClausesDisplay() {
    clausesContainer.innerHTML = '';
    const clauses = currentResults.clauses || [];
    
    const toShow = clauses.slice(0, visibleClauses);
    
    toShow.forEach((c, idx) => {
        const row = document.createElement('div');
        row.className = 'clause-row fade-in';
        
        const importance = c.importance || 'NORMAL';
        const badgeClass = importance === 'IMPORTANT' ? 'badge-IMPORTANT' : '';
        
        // Format similarity as percentage
        const simPercent = ((c.semantic_similarity || 0) * 100).toFixed(0) + "%";

        row.innerHTML = `
            <div class="clause-number">${idx + 1}</div>
            <div class="clause-card">
                <div class="clause-badge ${badgeClass}">${importance}</div>
                <div class="clause-split">
                    <div class="side-original">
                        <div class="clause-head">ORIGINAL</div>
                        <div class="clause-content">${c.original}</div>
                    </div>
                    <div class="side-implied">
                        <div class="clause-head">
                            SIMPLIFIED 
                            <span class="similarity-badge" title="Semantic Similarity Score">${simPercent} Match</span>
                        </div>
                        <div class="clause-content simplified">${c.simplified}</div>
                    </div>
                </div>
            </div>
        `;
        clausesContainer.appendChild(row);
    });

    // Pagination buttons
    if (clauses.length > 5) {
        paginationArea.classList.remove('hidden');
        if (visibleClauses < clauses.length) {
            loadMoreBtn.classList.remove('hidden');
            showLessBtn.classList.add('hidden');
        } else {
            loadMoreBtn.classList.add('hidden');
            showLessBtn.classList.remove('hidden');
        }
    } else {
        paginationArea.classList.add('hidden');
    }
}

loadMoreBtn.onclick = () => {
    visibleClauses = currentResults.clauses.length;
    updateClausesDisplay();
};

showLessBtn.onclick = () => {
    visibleClauses = 5;
    updateClausesDisplay();
    // Scroll back to top of clauses
    resultsDashboard.scrollIntoView({ behavior: 'smooth' });
};

function resetUI(clearActive = true) {
    if (clearActive) currentActiveId = null;
    resultsDashboard.classList.add('hidden');
    progressContainer.classList.add('hidden');
    fileReadyCard.classList.add('hidden');
    dropZone.classList.remove('hidden');
    startBtn.disabled = false;
    progressBar.style.width = "0%";
    if (clearActive) loadHistory();
}

// Download Btn (Client-side)
document.getElementById('download-btn').onclick = (e) => {
    e.preventDefault();
    if (!currentResults) return;
    
    const blob = new Blob([JSON.stringify(currentResults, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `LexiGen_Analysis_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
};
