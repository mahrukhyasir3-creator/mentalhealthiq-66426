const API_BASE_URL = 'http://localhost:8000';
const charts = {};

const state = {
    currentPage: 'home',
    predictions: [],
    fairnessReport: [],
    metrics: null,
    loading: false,
    error: null
};

function setActiveNav(page) {
    document.querySelectorAll('[data-page]').forEach(el => {
        if (el.dataset.page === page) {
            el.classList.add('active');
        } else {
            el.classList.remove('active');
        }
    });
}

async function apiCall(endpoint, method = 'GET', body = null) {
    const options = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) options.body = JSON.stringify(body);
    const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
    if (!response.ok) {
        const message = await handleApiError(response);
        throw new Error(message);
    }
    const text = await response.text();
    return text ? JSON.parse(text) : null;
}

async function handleApiError(response) {
    const text = await response.text();
    if (!text) return `HTTP ${response.status}`;
    try {
        const data = JSON.parse(text);
        if (data.detail) {
            return Array.isArray(data.detail) ? data.detail.map(item => item.msg || JSON.stringify(item)).join('; ') : data.detail;
        }
        if (data.message) return data.message;
        return JSON.stringify(data);
    } catch (error) {
        return text;
    }
}

function getHealth() {
    return apiCall('/health');
}

function predictAndSave(payload) {
    return apiCall('/predict-and-save', 'POST', payload);
}

function getPredictions(limit = 100) {
    return apiCall(`/predictions?limit=${limit}`);
}

function getFairnessReport() {
    return apiCall('/fairness-report');
}

function getMetrics() {
    return apiCall('/metrics').catch(() => null);
}

function navigate(page) {
    state.currentPage = page;
    state.error = null;
    setActiveNav(page);
    renderPage();
}

function renderPage() {
    const content = document.getElementById('content');
    content.innerHTML = '';
    switch (state.currentPage) {
        case 'dashboard':
            content.appendChild(createDashboardPage());
            loadDashboardData();
            break;
        case 'predict':
            content.appendChild(createPredictPage());
            break;
        case 'fairness':
            content.appendChild(createFairnessPage());
            loadFairnessData();
            break;
        case 'history':
            content.appendChild(createHistoryPage());
            loadHistoryData();
            break;
        case 'metrics':
            content.appendChild(createMetricsPage());
            loadMetricsData();
            break;
        default:
            content.appendChild(createHomePage());
    }
}

function createHomePage() {
    const div = document.createElement('div');
    div.innerHTML = `
        <div class="home-page">
            <div class="home-content">
                <h1 class="home-title">MentalHealthIQ</h1>
                <p class="home-description">Open the dashboard to explore predictions, fairness analysis, and model performance. Make a clinical-style depression risk prediction based on PHQ-9 responses and demographic data.</p>
                <div class="button-group">
                    <button class="btn btn-primary" onclick="navigate('dashboard')">Open Dashboard</button>
                    <button class="btn btn-secondary" onclick="navigate('predict')">Make Prediction</button>
                </div>
            </div>
        </div>
    `;
    return div;
}

function createDashboardPage() {
    const div = document.createElement('div');
    div.innerHTML = `
        <h2 class="page-title">Dashboard</h2>
        <div id="dashboard-content"><div class="loading">Loading dashboard...</div></div>
    `;
    return div;
}

function createPredictPage() {
    const div = document.createElement('div');
    div.innerHTML = `
        <h2 class="page-title">Predict Depression Severity</h2>
        <div class="form-section">
            <h3 class="page-subtitle">Patient Information</h3>
            <form id="prediction-form">
                <div class="form-row">
                    <div class="form-group"><label for="age">RIDAGEYR</label><input id="age" name="RIDAGEYR" type="number" min="18" max="100" value="30" required></div>
                    <div class="form-group"><label for="gender">RIAGENDR</label><input id="gender" name="RIAGENDR" type="number" min="1" max="9" value="1" required></div>
                    <div class="form-group"><label for="race">RIDRETH1</label><input id="race" name="RIDRETH1" type="number" min="1" max="9" value="1" required></div>
                </div>
                <div class="form-row">
                    <div class="form-group"><label for="income">INDHHIN2</label><input id="income" name="INDHHIN2" type="number" min="1" max="99" value="2" required></div>
                    <div class="form-group"><label for="education">DMDEDUC2</label><input id="education" name="DMDEDUC2" type="number" min="1" max="9" value="4" required></div>
                    <div class="form-group"><label for="marital">DMDMARTL</label><input id="marital" name="DMDMARTL" type="number" min="1" max="9" value="1" required></div>
                </div>
                <div class="form-group"><label for="age-group">AGE_GROUP</label><select id="age-group" name="AGE_GROUP" required><option>18-25</option><option selected>26-40</option><option>41-55</option><option>56-70</option><option>71+</option></select></div>
                <h3 class="page-subtitle">PHQ-9 Responses</h3>
                <p class="info-box">Use values 0-3 for DPQ010 through DPQ090.</p>
                <div class="dpq-grid">${['010','020','030','040','050','060','070','080','090'].map(code => `
                    <div class="form-group"><label for="DPQ${code}">DPQ${code}</label><input id="DPQ${code}" name="DPQ${code}" type="number" min="0" max="3" value="0" required></div>
                `).join('')}</div>
                <div class="form-actions">
                    <button class="btn btn-primary" type="submit" id="predict-submit">Submit & Save</button>
                    <span class="form-disclaimer">This prediction is not a medical diagnosis.</span>
                </div>
            </form>
            <div id="predict-message"></div>
        </div>
    `;
    div.querySelector('#prediction-form').addEventListener('submit', submitPrediction);
    return div;
}

function createHistoryPage() {
    const div = document.createElement('div');
    div.innerHTML = `
        <div class="history-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; flex-wrap:wrap; gap:1rem;">
                <div><h2 class="page-title">Prediction History</h2></div>
                <button class="btn btn-secondary" id="history-refresh">Refresh</button>
            </div>
            <div id="history-content"><div class="loading">Loading history...</div></div>
        </div>
    `;
    div.querySelector('#history-refresh').addEventListener('click', loadHistoryData);
    return div;
}

function createFairnessPage() {
    const div = document.createElement('div');
    div.innerHTML = `
        <h2 class="page-title">Fairness Analysis</h2>
        <div id="fairness-content"><div class="loading">Loading fairness report...</div></div>
    `;
    return div;
}

function createMetricsPage() {
    const div = document.createElement('div');
    div.innerHTML = `
        <h2 class="page-title">Metrics</h2>
        <div id="metrics-content"><div class="loading">Loading metrics...</div></div>
    `;
    return div;
}

function showError(container, message) {
    container.innerHTML = `<div class="error">${message}</div>`;
}

function renderStatusCard(label, value, subtitle = '') {
    return `<div class="stat-card"><div class="stat-label">${label}</div><div class="stat-value">${value}</div>${subtitle ? `<div style="margin-top:0.75rem; color:#475569; font-size:0.95rem;">${subtitle}</div>` : ''}</div>`;
}

async function loadDashboardData() {
    const container = document.getElementById('dashboard-content');
    container.innerHTML = '<div class="loading">Loading dashboard data...</div>';

    try {
        const health = await getHealth();
        const predictions = await getPredictions(100);
        const metrics = await getMetrics();

        const total = predictions.length;
        const avgRisk = total ? (predictions.reduce((sum, item) => sum + (item.risk_score || 0), 0) / total).toFixed(3) : '0.000';
        const highRisk = predictions.filter(item => (item.risk_score ?? 0) >= 0.7).length;
        const inferenceReady = health.status === 'healthy';
        const mongoReady = Array.isArray(predictions);

        const severityCounts = predictions.reduce((acc, item) => {
            const label = item.severity || 'Unknown';
            acc[label] = (acc[label] || 0) + 1;
            return acc;
        }, {});

        container.innerHTML = `
            <div class="stat-grid">
                ${renderStatusCard('API Status', inferenceReady ? 'Online' : 'Offline', `Model path: ${health.model_path || 'unknown'}`)}
                ${renderStatusCard('Inference Ready', inferenceReady ? 'Yes' : 'No')}
                ${renderStatusCard('Mongo Ready', mongoReady ? 'Yes' : 'No')}
            </div>
            <div class="stat-grid">
                ${renderStatusCard('Total Predictions', total)}
                ${renderStatusCard('Average Risk', avgRisk)}
                ${renderStatusCard('High Risk Count', highRisk, 'risk_score ≥ 0.70')}
            </div>
            <div class="chart-grid">
                <div class="chart-container chart-card"><h3 class="chart-title">Severity Distribution</h3><canvas id="severity-chart"></canvas></div>
                <div class="chart-container chart-card"><h3 class="chart-title">Recent Predictions</h3><div class="table-wrapper"><table><thead><tr><th>Timestamp</th><th>Severity</th><th>Risk</th></tr></thead><tbody>${predictions.slice(0, 5).map(item => `<tr><td>${new Date(item.timestamp).toLocaleString()}</td><td>${item.severity || '-'}</td><td>${item.risk_score ?? '-'}</td></tr>`).join('')}</tbody></table></div></div>
            </div>
        `;

        renderChart('severity-chart', Object.entries(severityCounts).map(([label, value]) => ({ label, value })), 'doughnut');

    } catch (error) {
        console.error(error);
        showError(container, `Dashboard failed: ${error.message}`);
    }
}

function renderChart(canvasId, data, type = 'bar') {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    if (charts[canvasId]) charts[canvasId].destroy();

    const labels = data.map(item => item.label);
    const values = data.map(item => item.value);
    const backgroundColor = ['#2563eb', '#1d4ed8', '#3b82f6', '#60a5fa', '#7dd3fc'];

    charts[canvasId] = new Chart(ctx, {
        type,
        data: {
            labels,
            datasets: [{
                label: 'Count',
                data: values,
                backgroundColor: backgroundColor.slice(0, values.length),
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: type === 'doughnut' },
                tooltip: { mode: 'index', intersect: false }
            }
        }
    });
}

async function submitPrediction(event) {
    event.preventDefault();
    const messageDiv = document.getElementById('predict-message');
    messageDiv.innerHTML = '';

    const age = Number(document.getElementById('age').value);
    const gender = Number(document.getElementById('gender').value);
    const race = Number(document.getElementById('race').value);
    const income = Number(document.getElementById('income').value);
    const education = Number(document.getElementById('education').value);
    const marital = Number(document.getElementById('marital').value);
    const ageGroup = document.getElementById('age-group').value;

    if (!age || age < 18 || age > 100) {
        showError(messageDiv, 'RIDAGEYR must be between 18 and 100.');
        return;
    }

    const payload = {
        RIDAGEYR: age,
        RIAGENDR: gender,
        RIDRETH1: race,
        INDHHIN2: income,
        DMDEDUC2: education,
        DMDMARTL: marital,
        AGE_GROUP: ageGroup
    };

    const dpqCodes = ['010','020','030','040','050','060','070','080','090'];
    for (const code of dpqCodes) {
        const field = document.getElementById(`DPQ${code}`);
        if (!field) {
            showError(messageDiv, `Missing field DPQ${code}.`);
            return;
        }
        const value = Number(field.value);
        if (field.value === '' || Number.isNaN(value)) {
            showError(messageDiv, `DPQ${code} is required.`);
            return;
        }
        if (value < 0 || value > 3) {
            showError(messageDiv, `DPQ${code} must be between 0 and 3.`);
            return;
        }
        payload[`DPQ${code}`] = value;
    }

    const requiredKeys = ['RIDAGEYR','RIAGENDR','RIDRETH1','INDHHIN2','DMDEDUC2','DMDMARTL','AGE_GROUP', ...dpqCodes.map(code => `DPQ${code}`)];
    for (const key of requiredKeys) {
        const value = payload[key];
        if (value === '' || value === null || value === undefined || (typeof value === 'number' && Number.isNaN(value))) {
            showError(messageDiv, `${key} is required.`);
            return;
        }
    }

    console.log('Payload:', payload);
    const button = document.getElementById('predict-submit');
    button.disabled = true;
    button.textContent = 'Predicting...';

    try {
        const response = await fetch(`${API_BASE_URL}/predict-and-save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const responseText = await response.clone().text();
        console.log(responseText);
        const data = response.ok ? JSON.parse(responseText) : null;

        if (!response.ok) {
            let message = responseText;
            try {
                const errorData = JSON.parse(responseText);
                if (errorData.detail) {
                    message = Array.isArray(errorData.detail) ? errorData.detail.map(item => item.msg || JSON.stringify(item)).join('; ') : errorData.detail;
                } else if (errorData.message) {
                    message = errorData.message;
                }
            } catch {
                message = responseText;
            }
            showError(messageDiv, message);
            return;
        }

        messageDiv.innerHTML = `
            <div class="success">
                <strong>Severity:</strong> ${data.severity}<br>
                <strong>Risk score:</strong> ${data.risk_score}<br>
                <strong>Model:</strong> ${data.model_type}<br>
                <strong>Saved:</strong> ${data.predictions_saved ? 'Yes' : 'No'}<br>
                ${data.warning ? `<span style="display:block; margin-top:0.75rem; color:#475569;">${data.warning}</span>` : ''}
            </div>
        `;
        document.getElementById('prediction-form').reset();
    } catch (error) {
        showError(messageDiv, `Prediction failed: ${error.message}`);
    } finally {
        button.disabled = false;
        button.textContent = 'Submit & Save';
    }
}

async function loadHistoryData() {
    const container = document.getElementById('history-content');
    container.innerHTML = '<div class="loading">Loading history...</div>';

    try {
        const data = await getPredictions(100);
        if (!Array.isArray(data) || data.length === 0) {
            container.innerHTML = '<div class="empty-state">No history available.</div>';
            return;
        }

        const rows = data.map(item => `
            <tr>
                <td>${new Date(item.timestamp).toLocaleString()}</td>
                <td>${item.input_data?.RIDAGEYR ?? '-'}</td>
                <td>${item.input_data?.RIAGENDR ?? '-'}</td>
                <td>${item.input_data?.AGE_GROUP ?? '-'}</td>
                <td>${item.severity ?? '-'}</td>
                <td>${item.risk_score ?? '-'}</td>
            </tr>
        `).join('');

        container.innerHTML = `
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr><th>Timestamp</th><th>Age</th><th>Gender</th><th>Age Group</th><th>Severity</th><th>Risk Score</th></tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        `;
    } catch (error) {
        showError(container, `History failed: ${error.message}`);
    }
}

function buildFairnessTable(records) {
    if (!records || !records.length) return '<div class="empty-state">No fairness records available.</div>';
    const rows = records.map(record => `
        <tr>
            <td>${record.Demographic || '-'}</td>
            <td>${record.Gender || record.Age_Group || record.Race || record.Income || '-'}</td>
            <td>${record.Accuracy ?? '-'}</td>
            <td>${record.FPR ?? record['FPR'] ?? '-'}</td>
            <td>${record.FNR ?? record['FNR'] ?? '-'}</td>
            <td>${record.Selection_Rate ?? record.Selection_Rate ?? '-'}</td>
        </tr>
    `).join('');
    return `
        <div class="table-wrapper">
            <table>
                <thead><tr><th>Demographic</th><th>Group</th><th>Accuracy</th><th>FPR</th><th>FNR</th><th>Selection Rate</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    `;
}

function normalizeFairnessRecords(reportData) {
    if (!reportData || !Array.isArray(reportData.records)) return [];
    return reportData.records.map(record => ({
        Demographic: record.Demographic,
        Group: record.Gender || record.Age_Group || record.Race || record.Income || '-',
        Accuracy: Number(record.Accuracy ?? 0),
        FPR: Number(record.FPR ?? 0),
        FNR: Number(record.FNR ?? 0),
        Selection_Rate: Number(record.Selection_Rate ?? 0)
    }));
}

function renderFairnessChart(canvasId, labels, values, label) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    if (charts[canvasId]) charts[canvasId].destroy();
    charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{ label, data: values, backgroundColor: '#2563eb' }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { beginAtZero: true } }
        }
    });
}

async function loadFairnessData() {
    const container = document.getElementById('fairness-content');
    container.innerHTML = '<div class="loading">Loading fairness report...</div>';

    try {
        const reportData = await getFairnessReport();
        const records = normalizeFairnessRecords(reportData);
        if (!records.length) {
            container.innerHTML = '<div class="empty-state">No fairness data available.</div>';
            return;
        }

        const accuracyLabels = records.map(r => `${r.Demographic}: ${r.Group}`);
        const accuracyValues = records.map(r => r.Accuracy);
        const fprValues = records.map(r => r.FPR);
        const fnrValues = records.map(r => r.FNR);
        const selectionValues = records.map(r => r.Selection_Rate);

        container.innerHTML = `
            <div class="chart-grid">
                <div class="chart-container chart-card"><h3 class="chart-title">Accuracy by Group</h3><canvas id="fairness-accuracy"></canvas></div>
                <div class="chart-container chart-card"><h3 class="chart-title">False Positive Rate</h3><canvas id="fairness-fpr"></canvas></div>
                <div class="chart-container chart-card"><h3 class="chart-title">False Negative Rate</h3><canvas id="fairness-fnr"></canvas></div>
                <div class="chart-container chart-card"><h3 class="chart-title">Selection Rate</h3><canvas id="fairness-selection"></canvas></div>
            </div>
            <div class="history-card"><h3 class="page-subtitle">Fairness Records</h3>${buildFairnessTable(records)}</div>
        `;

        renderFairnessChart('fairness-accuracy', accuracyLabels, accuracyValues, 'Accuracy');
        renderFairnessChart('fairness-fpr', accuracyLabels, fprValues, 'FPR');
        renderFairnessChart('fairness-fnr', accuracyLabels, fnrValues, 'FNR');
        renderFairnessChart('fairness-selection', accuracyLabels, selectionValues, 'Selection Rate');
    } catch (error) {
        showError(container, `Fairness failed: ${error.message}`);
    }
}

function renderMetricsCards(content, metrics) {
    const data = [
        { label: 'Accuracy', value: metrics.accuracy ?? 'N/A' },
        { label: 'Precision', value: metrics.precision ?? 'N/A' },
        { label: 'Recall', value: metrics.recall ?? 'N/A' },
        { label: 'F1 Score', value: metrics.f1 ?? 'N/A' }
    ];

    const cards = data.map(item => `
        <div class="metrics-card"><div class="stat-label">${item.label}</div><div class="stat-value">${item.value}</div></div>
    `).join('');

    content.innerHTML = `
        <div class="metrics-grid">${cards}</div>
        <div class="history-card"><h3 class="page-subtitle">Confusion Matrix</h3>${renderConfusionMatrix(metrics.confusion_matrix)}</div>
    `;
}

function renderConfusionMatrix(matrix) {
    if (!matrix || !Array.isArray(matrix)) return '<div class="empty-state">Confusion matrix not available.</div>';
    const rows = matrix.map((row, idx) => `<tr><td>Actual ${idx}</td>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('');
    return `
        <div class="table-wrapper">
            <table>
                <thead><tr><th></th>${matrix[0].map((_, idx) => `<th>Pred ${idx}</th>`).join('')}</tr></thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    `;
}

async function loadMetricsData() {
    const container = document.getElementById('metrics-content');
    container.innerHTML = '<div class="loading">Loading metrics...</div>';

    try {
        const data = await getMetrics();
        if (!data || !data.metrics) {
            showError(container, 'Metrics endpoint unavailable.');
            return;
        }
        renderMetricsCards(container, data.metrics);
    } catch (error) {
        showError(container, `Metrics failed: ${error.message}`);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const initialPage = 'home';
    setActiveNav(initialPage);
    renderPage();
});
