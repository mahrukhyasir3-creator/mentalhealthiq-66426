const API_BASE_URL = window.API_BASE_URL || 'http://localhost:8000';
const charts = {};

const PHQ_QUESTIONS = [
    ['DPQ010', 'Little interest or pleasure in doing things'],
    ['DPQ020', 'Feeling down, depressed, or hopeless'],
    ['DPQ030', 'Trouble falling or staying asleep, or sleeping too much'],
    ['DPQ040', 'Feeling tired or having little energy'],
    ['DPQ050', 'Poor appetite or overeating'],
    ['DPQ060', 'Feeling bad about yourself'],
    ['DPQ070', 'Trouble concentrating'],
    ['DPQ080', 'Moving or speaking slowly, or being fidgety/restless'],
    ['DPQ090', 'Thoughts that you would be better off dead or hurting yourself']
];

const PHQ_OPTIONS = [
    [0, 'Not at all'],
    [1, 'Several days'],
    [2, 'More than half the days'],
    [3, 'Nearly every day']
];

const SEVERITIES = ['Minimal', 'Mild', 'Moderate', 'Moderately Severe', 'Severe'];

const state = {
    page: 'home',
    fairnessTab: 'AGE_GROUP',
    history: [],
    fairnessReport: [],
    fairnessSummary: null,
    patientOptions: [],
    patientOptionRecords: [],
    comparisonPatients: [],
    reportPatients: [],
    filteredHistory: [],
    lastPayload: readStorage('mhiq_last_payload'),
    lastResult: readStorage('mhiq_last_result')
};

function readStorage(key) {
    try {
        return JSON.parse(localStorage.getItem(key) || 'null');
    } catch {
        return null;
    }
}

function writeStorage(key, value) {
    localStorage.setItem(key, JSON.stringify(value));
}

async function apiCall(endpoint, method = 'GET', body = null) {
    const options = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) options.body = JSON.stringify(body);

    const response = await fetch(`${API_BASE_URL}${endpoint}`, options);
    const text = await response.text();
    let data = null;

    try {
        data = text ? JSON.parse(text) : null;
    } catch {
        data = text;
    }

    if (!response.ok) {
        const detail = data?.detail || data?.message || data || `HTTP ${response.status}`;
        throw new Error(Array.isArray(detail) ? detail.map(item => item.msg || JSON.stringify(item)).join('; ') : detail);
    }

    return data;
}

function navigateTo(page) {
    state.page = page;
    setActiveNav(page);
    destroyCharts();
    renderPage();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

const navigate = navigateTo;

function setActiveNav(page) {
    document.querySelectorAll('[data-page]').forEach(link => {
        link.classList.toggle('active', link.dataset.page === page);
    });
}

function showLoading(target, message = 'Loading...') {
    const element = resolveTarget(target);
    if (element) {
        element.className = 'loading-state';
        element.innerHTML = message;
    }
}

function showError(target, message) {
    const element = resolveTarget(target);
    if (element) {
        element.className = 'error-state';
        element.innerHTML = `<strong>Something went wrong.</strong><br>${escapeHtml(message)}`;
    }
}

function showEmpty(target, message) {
    const element = resolveTarget(target);
    if (element) {
        element.className = 'empty-state';
        element.innerHTML = escapeHtml(message);
    }
}

function resolveTarget(target) {
    if (typeof target === 'string') return document.getElementById(target);
    return target;
}

function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function renderPage() {
    const content = document.getElementById('content');
    content.innerHTML = '';
    const page = document.createElement('section');
    page.className = 'page';
    content.appendChild(page);

    const renderers = {
        home: renderHomePage,
        dashboard: renderDashboardPage,
        predict: renderPredictPage,
        result: renderResultPage,
        fairness: renderFairnessPage,
        history: renderHistoryPage,
        comparison: renderComparisonPage,
        reports: renderReportsPage,
        metrics: renderMetricsPage,
        health: renderHealthPage
    };

    (renderers[state.page] || renderHomePage)(page);
}

function pageHeader(eyebrow, title, subtitle, actions = '') {
    return `
        <header class="page-header">
            <div>
                <div class="eyebrow">${eyebrow}</div>
                <h1 class="page-title">${title}</h1>
                <p class="page-subtitle">${subtitle}</p>
            </div>
            ${actions ? `<div class="button-row no-print">${actions}</div>` : ''}
        </header>
    `;
}

function renderHomePage(page) {
    page.innerHTML = `
        <div class="home-hero">
            <section class="hero-panel">
                <div>
                    <div class="eyebrow">Clinical depression screening</div>
                    <h1 class="home-title">MentalHealthIQ</h1>
                    <p class="home-description">PHQ-9 Depression Severity Screening for clinics. Staff fill a simple patient form, the API predicts depression severity, and doctors get risk, recommendation, fairness notes, history, comparison, and printable reports.</p>
                    <div class="button-row">
                        <button class="btn btn-primary" onclick="navigateTo('predict')">Start PHQ-9 Form</button>
                        <button class="btn btn-secondary" onclick="navigateTo('dashboard')">Open Dashboard</button>
                    </div>
                </div>
            </section>
            <aside class="hero-side">
                <div class="soft-card">
                    <span class="badge badge-low">Real dataset</span>
                    <h2 class="section-title" style="margin-top:1rem;">NHANES CDC data</h2>
                    <p class="section-note">The model pipeline uses real demographic and questionnaire CSV files from the National Health and Nutrition Examination Survey.</p>
                </div>
                <div class="feature-grid">
                    ${featureCard('PHQ-9 screening', 'All 9 questions, live score, severity, and risk band.')}
                    ${featureCard('Patient history', 'Saved visits, filters, and patient comparison when MongoDB is configured.')}
                    ${featureCard('Fairness analysis', 'Age, gender, race, and income group risk comparisons.')}
                    ${featureCard('Reports', 'Doctor-friendly printable report for each prediction.')}
                </div>
            </aside>
        </div>
    `;
}

function featureCard(title, text) {
    return `<div class="soft-card"><h3 class="section-title">${title}</h3><p class="section-note">${text}</p></div>`;
}

function renderDashboardPage(page) {
    page.innerHTML = `
        ${pageHeader('Clinic overview', 'Dashboard', 'Saved prediction trends and model status. If MongoDB is not configured, dashboard still shows model and fairness status.')}
        <div id="dashboard-content" class="main-content-card"></div>
    `;
    loadDashboard();
}

async function loadDashboard() {
    showLoading('dashboard-content', 'Loading dashboard...');
    try {
        const [health, stats, predictions, fairness] = await Promise.all([
            apiCall('/health'),
            apiCall('/stats').catch(() => null),
            apiCall('/predictions?limit=500').catch(() => []),
            apiCall('/fairness-report').catch(() => ({ records: [] }))
        ]);

        state.history = Array.isArray(predictions) ? predictions : [];
        state.fairnessReport = fairness.records || [];
        const dashboardStats = stats || buildStats(state.history);
        const averagePhq = average(state.history.map(item => Number(item.phq9_total || 0)).filter(Boolean));
        const hasPredictions = state.history.length > 0;

        const container = document.getElementById('dashboard-content');
        container.className = '';
        container.innerHTML = `
            <div class="stat-grid">
                ${statCard('Total predictions', dashboardStats.total_predictions || 0)}
                ${statCard('High risk patients', dashboardStats.high_risk_patients || 0, 'Urgent follow-up')}
                ${statCard('Medium risk patients', dashboardStats.medium_risk_patients || 0)}
                ${statCard('Low risk patients', dashboardStats.low_risk_patients || 0)}
                ${statCard('Average PHQ-9 score', averagePhq ? averagePhq.toFixed(1) : '0.0')}
                ${statCard('Average risk score', percent(dashboardStats.average_risk_score || 0))}
                ${statCard('Latest prediction time', dashboardStats.latest_prediction_time ? formatDateTime(dashboardStats.latest_prediction_time) : 'No saved visits')}
                ${statCard('Model status', health.model || 'unknown')}
            </div>
            ${!hasPredictions ? `<div class="empty-state" style="margin-top:1rem;">No predictions saved yet. Submit a patient form and configure MongoDB to populate dashboard history.</div>` : ''}
            <div class="chart-grid" style="margin-top:1rem;">
                ${chartCard('severityChart', 'Severity distribution')}
                ${chartCard('riskBandChart', 'Risk band distribution')}
                ${chartCard('predictionsTimeChart', 'Predictions over time')}
                ${chartCard('phqAverageChart', 'Average PHQ-9 score over time')}
                ${chartCard('genderRiskChart', 'Gender-wise risk')}
                ${chartCard('ageRiskChart', 'Age-wise risk')}
            </div>
            <div class="card" style="margin-top:1rem;">
                <h2 class="section-title">Recent patient predictions</h2>
                ${renderHistoryRecords(state.history.slice(0, 8))}
            </div>
        `;

        renderSeverityChart(dashboardStats.severity_counts || {});
        renderRiskBandChart(dashboardStats.risk_band_counts || {});
        renderTimeCharts(state.history);
        renderFairnessDashboardCharts(state.fairnessReport);
    } catch (error) {
        showError('dashboard-content', `Dashboard failed: ${error.message}`);
    }
}

function buildStats(records) {
    const risk_band_counts = { Low: 0, Medium: 0, High: 0 };
    const severity_counts = {};
    let riskTotal = 0;

    records.forEach(item => {
        const severity = item.predicted_severity || item.severity || 'Unknown';
        const riskBand = item.risk_band || getRiskBandFromSeverity(severity);
        risk_band_counts[riskBand] = (risk_band_counts[riskBand] || 0) + 1;
        severity_counts[severity] = (severity_counts[severity] || 0) + 1;
        riskTotal += Number(item.risk_score || 0);
    });

    return {
        total_predictions: records.length,
        high_risk_patients: risk_band_counts.High || 0,
        medium_risk_patients: risk_band_counts.Medium || 0,
        low_risk_patients: risk_band_counts.Low || 0,
        average_risk_score: records.length ? riskTotal / records.length : 0,
        latest_prediction_time: records[0]?.timestamp,
        severity_counts,
        risk_band_counts
    };
}

function renderPredictPage(page) {
    page.innerHTML = `
        ${pageHeader('Patient screening', 'PHQ-9 Patient Form', 'Fill patient details and all 9 PHQ-9 questions. You can predict only, or predict and save to MongoDB.')}
        ${stepper(1)}
        <form id="prediction-form" class="form-card" novalidate>
            <h2 class="section-title">Patient details</h2>
            <p class="section-note">These fields make reports, saved history, and patient comparison easier for clinic staff.</p>
            <div class="form-grid">
                ${patientIdField()}
                ${inputField('Patient Name', 'patient_name', 'text', '', 'required')}
                ${inputField('Age', 'RIDAGEYR', 'number', '35', 'required min="0" max="120"')}
                ${selectField('Gender', 'RIAGENDR', [['1', 'Male'], ['2', 'Female']], '2')}
                ${selectField('Race / Ethnicity', 'RIDRETH1', [['1', 'Mexican American'], ['2', 'Other Hispanic'], ['3', 'Non-Hispanic White'], ['4', 'Non-Hispanic Black'], ['5', 'Other Race']], '3')}
                ${selectField('Income Group', 'INDHHIN2', [['1', 'Low income'], ['2', 'Lower middle'], ['3', 'Middle'], ['4', 'Upper middle'], ['5', 'High income'], ['77', 'Refused'], ['99', 'Unknown']], '3')}
                ${selectField('Education Level', 'DMDEDUC2', [['1', 'Less than 9th grade'], ['2', '9th to 11th grade'], ['3', 'High school'], ['4', 'Some college'], ['5', 'College graduate']], '4')}
                ${selectField('Marital Status', 'DMDMARTL', [['1', 'Married'], ['2', 'Widowed'], ['3', 'Divorced'], ['4', 'Separated'], ['5', 'Never married'], ['6', 'Living with partner']], '1')}
                ${inputField('Visit Date', 'visit_date', 'date', today(), 'required')}
                ${inputField('Visit Time', 'visit_time', 'time', currentTime(), 'required')}
            </div>
            <div class="form-group" style="margin-top:1rem;">
                <label for="doctor_notes">Doctor Notes</label>
                <textarea id="doctor_notes" name="doctor_notes" placeholder="Optional notes for the printable report"></textarea>
            </div>

            <div class="card" style="box-shadow:none; margin:1.25rem 0;">
                <div class="form-grid">
                    <div>
                        <h3 class="section-title">Form completion</h3>
                        <p class="section-note"><span id="completionText">0%</span> complete</p>
                        <div class="progress-track"><div id="completionBar" class="progress-fill low"></div></div>
                    </div>
                    <div>
                        <h3 class="section-title">Live PHQ-9 score</h3>
                        <p class="section-note"><strong id="phqTotalText">0</strong> / 27 - <span id="severityText">Minimal</span> - <span id="riskText">Low risk</span></p>
                        <div class="progress-track"><div id="phqBar" class="progress-fill low"></div></div>
                    </div>
                </div>
            </div>

            <h2 class="section-title">PHQ-9 questionnaire</h2>
            <p class="section-note">Choose the answer that best describes the last two weeks.</p>
            <div class="phq-list">
                ${PHQ_QUESTIONS.map(([code, question], index) => phqQuestionCard(code, question, index)).join('')}
            </div>

            <div id="form-message" style="margin-top:1rem;"></div>
            <div class="button-row" style="margin-top:1.25rem;">
                <button class="btn btn-secondary" type="button" onclick="calculateLocalPreview()">Calculate Locally</button>
                <button class="btn btn-primary" type="button" onclick="submitPrediction(false)">Send to API</button>
                <button class="btn btn-primary" type="button" onclick="submitPrediction(true)">Predict and Save</button>
                <button class="btn btn-secondary" type="button" onclick="resetPredictionForm()">Reset Form</button>
            </div>
        </form>
    `;

    const form = document.getElementById('prediction-form');
    form.addEventListener('input', calculateLocalPreview);
    form.addEventListener('change', event => {
        if (event.target?.id === 'patient_id') applySelectedPatient();
    });
    loadPatientOptions();
    calculateLocalPreview();
}

function stepper(activeStep) {
    const steps = ['Fill Form', 'Send to API', 'View Result', 'Save / Generate Report'];
    return `<div class="stepper no-print">${steps.map((label, index) => `<div class="step ${index + 1 <= activeStep ? 'active' : ''}"><span class="step-number">${index + 1}</span><span>${label}</span></div>`).join('')}</div>`;
}

function inputField(label, name, type, value = '', attrs = '') {
    return `<div class="form-group"><label for="${name}">${label}</label><input id="${name}" name="${name}" type="${type}" value="${escapeHtml(value)}" ${attrs}></div>`;
}

function patientIdField() {
    return `
        <div class="form-group">
            <label for="patient_id">Patient ID</label>
            <div class="input-action-row">
                <input id="patient_id" name="patient_id" type="text" list="patient-options" placeholder="Auto assigned if empty">
                <button class="btn btn-secondary" type="button" onclick="assignPatientId()">Auto ID</button>
            </div>
            <datalist id="patient-options"></datalist>
            <small class="field-help">Use an existing ID to add another visit to that patient's history.</small>
        </div>
    `;
}

function selectField(label, name, options, selected = '') {
    return `<div class="form-group"><label for="${name}">${label}</label><select id="${name}" name="${name}" required>${options.map(([value, text]) => `<option value="${value}" ${String(value) === String(selected) ? 'selected' : ''}>${text}</option>`).join('')}</select></div>`;
}

function phqQuestionCard(code, question, index) {
    return `
        <div class="phq-item">
            <div>
                <div class="phq-code">${code}</div>
                <strong>${index + 1}. ${question}</strong>
            </div>
            <div class="radio-group" role="radiogroup" aria-label="${code}">
                ${PHQ_OPTIONS.map(([value, label]) => `
                    <label class="radio-option">
                        <input type="radio" name="${code}" value="${value}" ${value === 0 ? 'checked' : ''} required>
                        <span>${value} - ${label}</span>
                    </label>
                `).join('')}
            </div>
        </div>
    `;
}

function calculatePHQ9Total() {
    return PHQ_QUESTIONS.reduce((sum, [code]) => sum + Number(document.querySelector(`input[name="${code}"]:checked`)?.value ?? 0), 0);
}

function getSeverityFromPHQ9(total) {
    if (total <= 4) return 'Minimal';
    if (total <= 9) return 'Mild';
    if (total <= 14) return 'Moderate';
    if (total <= 19) return 'Moderately Severe';
    return 'Severe';
}

function getRiskBandFromSeverity(severity) {
    if (severity === 'Moderate') return 'Medium';
    if (severity === 'Moderately Severe' || severity === 'Severe') return 'High';
    return 'Low';
}

const getRiskBand = getRiskBandFromSeverity;

function deriveAgeGroup(age) {
    const numericAge = Number(age);
    if (Number.isNaN(numericAge)) return '';
    if (numericAge < 18) return 'Under 18';
    if (numericAge <= 29) return '18-29';
    if (numericAge <= 44) return '30-44';
    if (numericAge <= 59) return '45-59';
    return '60+';
}

function stablePatientHash(value) {
    let hash = 0;
    String(value).split('').forEach(character => {
        hash = ((hash << 5) - hash + character.charCodeAt(0)) >>> 0;
    });
    return hash.toString(36).toUpperCase().padStart(6, '0').slice(-6);
}

function generatePatientIdFromValues(name, age, gender) {
    const slug = String(name || 'Patient')
        .trim()
        .replace(/[^A-Za-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '')
        .toUpperCase()
        .slice(0, 18) || 'PATIENT';
    return `MHQ-${slug}-${stablePatientHash(`${slug}|${Number(age) || 0}|${Number(gender) || 0}`)}`;
}

function generatePatientIdFromForm() {
    return generatePatientIdFromValues(
        valueOf('patient_name'),
        valueOf('RIDAGEYR'),
        valueOf('RIAGENDR')
    );
}

function assignPatientId() {
    const patientIdInput = document.getElementById('patient_id');
    if (!patientIdInput) return '';
    patientIdInput.value = generatePatientIdFromForm();
    const message = document.getElementById('form-message');
    if (message) {
        message.className = 'success';
        message.innerHTML = `Patient ID assigned: <strong>${escapeHtml(patientIdInput.value)}</strong>`;
    }
    calculateLocalPreview();
    return patientIdInput.value;
}

async function loadPatientOptions() {
    const dataList = document.getElementById('patient-options');
    if (!dataList) return;

    try {
        const response = await apiCall('/predictions?limit=500').catch(() => []);
        const records = normalizePredictionsResponse(response);
        state.patientOptionRecords = records;
        state.patientOptions = getUniquePatients(records);
        dataList.innerHTML = state.patientOptions.map(patient => {
            return `<option value="${escapeHtml(patient.patient_id)}" label="${escapeHtml(patient.label)}"></option>`;
        }).join('');
    } catch {
        state.patientOptions = [];
        state.patientOptionRecords = [];
    }
}

function normalizePredictionsResponse(response) {
    if (Array.isArray(response)) return response;
    if (Array.isArray(response?.predictions)) return response.predictions;
    if (Array.isArray(response?.data)) return response.data;
    if (Array.isArray(response?.items)) return response.items;
    if (Array.isArray(response?.results)) return response.results;
    return [];
}

const predictionRecordsFromResponse = normalizePredictionsResponse;

async function fetchSavedPatients() {
    const response = await apiCall('/predictions?limit=500');
    const records = normalizePredictionsResponse(response);
    if (!records.length) await ensureSavedPatientStoreAvailable();
    return getUniquePatients(records);
}

async function ensureSavedPatientStoreAvailable() {
    const health = await apiCall('/health').catch(() => null);
    if (health && String(health.mongo || '').toLowerCase() !== 'ready') {
        throw new Error('MongoDB is unavailable.');
    }
}

function getUniquePatients(records) {
    const byPatient = new Map();
    records.forEach(record => {
        const patientId = getPatientId(record);
        if (!patientId) return;

        const existing = byPatient.get(patientId);
        const patientName = getPatientName(record) || existing?.patient_name || '';
        const latestVisit = latestVisitValue(existing?.latest_visit, getVisitValue(record));
        byPatient.set(patientId, {
            patient_id: patientId,
            patient_name: patientName,
            latest_visit: latestVisit,
            visit_count: (existing?.visit_count || 0) + 1,
            label: ''
        });
    });

    return [...byPatient.values()]
        .map(patient => ({
            ...patient,
            label: patientLabel(patient)
        }))
        .sort((a, b) => {
            const nameCompare = (a.patient_name || '').localeCompare(b.patient_name || '');
            return nameCompare || a.patient_id.localeCompare(b.patient_id);
        });
}

const uniquePatientsFromPredictions = getUniquePatients;

function getPatientId(record) {
    return String(
        record?.patient_id
        || record?.patientId
        || record?.id
        || record?.input_data?.patient_id
        || record?.input_data?.patientId
        || record?.input_data?.id
        || ''
    ).trim();
}

function getPatientName(record) {
    return String(
        record?.patient_name
        || record?.patientName
        || record?.name
        || record?.input_data?.patient_name
        || record?.input_data?.patientName
        || record?.input_data?.name
        || ''
    ).trim();
}

function getVisitValue(record) {
    return record?.visit_date || record?.timestamp || record?.created_at || record?.input_data?.visit_date || '';
}

function latestVisitValue(current, candidate) {
    if (!current) return candidate || '';
    if (!candidate) return current;
    return String(candidate) > String(current) ? candidate : current;
}

function patientLabel(patient) {
    const base = patient.patient_name ? `${patient.patient_name} - ${patient.patient_id}` : patient.patient_id;
    return patient.visit_count ? `${base} (${patient.visit_count} ${patient.visit_count === 1 ? 'visit' : 'visits'})` : base;
}

function patientSearchField(label, inputId, datalistId) {
    return `
        <div class="form-group">
            <label for="${inputId}">${label}</label>
            <input id="${inputId}" type="text" list="${datalistId}" placeholder="Search or select a patient" autocomplete="off">
            <datalist id="${datalistId}"></datalist>
        </div>
    `;
}

async function populatePatientSearchDropdown(inputId, datalistId, statusId = null, records = null) {
    const input = document.getElementById(inputId);
    const datalist = document.getElementById(datalistId);
    const status = statusId ? document.getElementById(statusId) : null;
    if (!input || !datalist) return [];

    input.dataset.patientLoadFailed = 'false';
    if (status) showLoading(status, 'Loading saved patients...');

    try {
        if (records && !records.length) await ensureSavedPatientStoreAvailable();
        const patients = records ? getUniquePatients(records) : await fetchSavedPatients();
        datalist.innerHTML = patients.map(patient => {
            return `<option value="${escapeHtml(patient.patient_id)}" label="${escapeHtml(patient.label)}"></option>`;
        }).join('');

        input.placeholder = patients.length ? 'Search or select a patient' : 'No saved patients found';

        if (status) {
            if (patients.length) {
                status.className = 'success';
                status.innerHTML = `Loaded ${patients.length} saved patient${patients.length === 1 ? '' : 's'}.`;
            } else {
                showEmpty(status, 'No saved patients found. Use Predict & Save first.');
            }
        }

        return patients;
    } catch (error) {
        input.dataset.patientLoadFailed = 'true';
        datalist.innerHTML = '';
        input.placeholder = 'Saved patients unavailable';
        if (status) {
            showError(status, 'Could not load saved patients. Make sure API and MongoDB are running.');
        }
        return [];
    }
}

function getSelectedPatientId(inputId) {
    const rawValue = valueOf(inputId);
    if (!rawValue) return '';

    const allPatients = [
        ...state.patientOptions,
        ...state.comparisonPatients,
        ...state.reportPatients,
        ...getUniquePatients(state.history)
    ];
    const selected = allPatients.find(patient => {
        return patient.patient_id === rawValue || patient.label === rawValue;
    });
    return selected?.patient_id || rawValue;
}

function applySelectedPatient() {
    const patientId = valueOf('patient_id');
    if (!patientId) return;

    const selected = state.patientOptionRecords.find(record => {
        return getPatientId(record) === patientId;
    });
    if (!selected) return;

    const source = selected.input_data || selected;
    setInputValue('patient_name', getPatientName(selected));
    setInputValue('RIDAGEYR', source.RIDAGEYR);
    setInputValue('RIAGENDR', source.RIAGENDR);
    setInputValue('RIDRETH1', source.RIDRETH1);
    setInputValue('INDHHIN2', source.INDHHIN2);
    setInputValue('DMDEDUC2', source.DMDEDUC2);
    setInputValue('DMDMARTL', source.DMDMARTL);
    calculateLocalPreview();
}

function calculateLocalPreview() {
    const total = calculatePHQ9Total();
    const severity = getSeverityFromPHQ9(total);
    const riskBand = getRiskBandFromSeverity(severity);
    const form = document.getElementById('prediction-form');
    const fields = form ? [...form.querySelectorAll('input[required], select[required]')] : [];
    const filled = fields.filter(field => field.type === 'radio' ? form.querySelector(`input[name="${field.name}"]:checked`) : field.value).length;
    const uniqueRequired = new Set(fields.map(field => field.name || field.id));
    const completion = uniqueRequired.size ? Math.round((new Set(fields.filter(field => field.type === 'radio' ? form.querySelector(`input[name="${field.name}"]:checked`) : field.value).map(field => field.name || field.id)).size / uniqueRequired.size) * 100) : 0;

    setText('completionText', `${completion}%`);
    setText('phqTotalText', total);
    setText('severityText', severity);
    setText('riskText', `${riskBand} risk`);
    setProgress('completionBar', completion, 'low');
    setProgress('phqBar', Math.round((total / 27) * 100), riskBand.toLowerCase());
}

function setText(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
}

function setProgress(id, value, riskClass) {
    const element = document.getElementById(id);
    if (!element) return;
    element.style.width = `${Math.max(0, Math.min(100, Number(value) || 0))}%`;
    element.className = `progress-fill ${riskClass || ''}`;
}

function collectPatientPayload() {
    const form = document.getElementById('prediction-form');
    const data = new FormData(form);
    const age = Number(data.get('RIDAGEYR'));
    const patientId = String(data.get('patient_id') || '').trim()
        || generatePatientIdFromValues(data.get('patient_name'), age, data.get('RIAGENDR'));
    const patientIdInput = document.getElementById('patient_id');
    if (patientIdInput && !patientIdInput.value.trim()) {
        patientIdInput.value = patientId;
    }
    const payload = {
        patient_id: patientId,
        patient_name: String(data.get('patient_name') || '').trim(),
        visit_date: String(data.get('visit_date') || '').trim(),
        visit_time: String(data.get('visit_time') || '').trim(),
        doctor_notes: String(data.get('doctor_notes') || '').trim(),
        RIDAGEYR: age,
        RIAGENDR: Number(data.get('RIAGENDR')),
        RIDRETH1: Number(data.get('RIDRETH1')),
        INDHHIN2: Number(data.get('INDHHIN2')),
        DMDEDUC2: Number(data.get('DMDEDUC2')),
        DMDMARTL: Number(data.get('DMDMARTL')),
        AGE_GROUP: deriveAgeGroup(age)
    };

    PHQ_QUESTIONS.forEach(([code]) => {
        payload[code] = Number(data.get(code));
    });

    return payload;
}

function validatePredictionForm(payload) {
    const errors = [];
    if (!payload.patient_name) errors.push('Patient name is required.');
    if (!Number.isFinite(payload.RIDAGEYR) || payload.RIDAGEYR < 0 || payload.RIDAGEYR > 120) errors.push('Age must be between 0 and 120.');
    if (!payload.visit_date) errors.push('Visit date is required.');
    if (!payload.visit_time) errors.push('Visit time is required.');
    PHQ_QUESTIONS.forEach(([code]) => {
        if (![0, 1, 2, 3].includes(payload[code])) errors.push(`${code} must be answered from 0 to 3.`);
    });
    return errors;
}

async function submitPrediction(save = false) {
    const message = document.getElementById('form-message');
    const payload = collectPatientPayload();
    const errors = validatePredictionForm(payload);
    if (errors.length) {
        showError(message, errors.join('<br>'));
        return;
    }

    showLoading(message, save ? 'Sending to API and saving report...' : 'Sending to API...');

    try {
        let result;
        if (save) {
            try {
                result = await apiCall('/predict-and-save', 'POST', payload);
            } catch (error) {
                const message = String(error.message);
                if (!message.includes('MongoDB repository is not configured') && !message.includes('MongoDB is not available')) {
                    throw error;
                }
                result = await apiCall('/predict', 'POST', payload);
                result.predictions_saved = false;
                result.mongo_note = 'MongoDB is unavailable, so this prediction was not saved.';
            }
        } else {
            result = await apiCall('/predict', 'POST', payload);
        }

        state.lastPayload = payload;
        state.lastResult = normalizeResult(result, payload);
        writeStorage('mhiq_last_payload', state.lastPayload);
        writeStorage('mhiq_last_result', state.lastResult);
        navigateTo('result');
    } catch (error) {
        showError(message, `Prediction failed: ${error.message}`);
    }
}

function normalizeResult(result, payload) {
    const severity = result.predicted_severity || result.severity || getSeverityFromPHQ9(calculatePHQ9TotalFromPayload(payload));
    const riskBand = result.risk_band || getRiskBandFromSeverity(severity);
    const phq9Total = result.phq9_total ?? calculatePHQ9TotalFromPayload(payload);
    return {
        ...result,
        patient_id: result.patient_id || payload.patient_id,
        patient_name: result.patient_name || payload.patient_name,
        visit_date: result.visit_date || payload.visit_date,
        visit_time: result.visit_time || payload.visit_time,
        doctor_notes: result.doctor_notes || payload.doctor_notes,
        predicted_severity: severity,
        severity,
        risk_band: riskBand,
        risk_score: Number(result.risk_score || 0),
        risk_percentage: result.risk_percentage ?? Math.round(Number(result.risk_score || 0) * 100),
        phq9_total: phq9Total,
        recommendation: result.recommendation || recommendationForSeverity(severity),
        warning: result.warning || warningForSeverity(severity),
        probabilities: result.probabilities || {},
        timestamp: result.timestamp || new Date().toISOString()
    };
}

function calculatePHQ9TotalFromPayload(payload) {
    return PHQ_QUESTIONS.reduce((sum, [code]) => sum + Number(payload?.[code] || 0), 0);
}

function resetPredictionForm() {
    const form = document.getElementById('prediction-form');
    if (!form) return;
    form.reset();
    PHQ_QUESTIONS.forEach(([code]) => {
        const first = form.querySelector(`input[name="${code}"][value="0"]`);
        if (first) first.checked = true;
    });
    calculateLocalPreview();
    showEmpty('form-message', 'Form reset. Fill patient details and PHQ-9 answers.');
}

function renderResultPage(page) {
    if (!state.lastResult) {
        page.innerHTML = `
            ${pageHeader('Prediction result', 'No Result Yet', 'Submit a patient PHQ-9 form to view the result here.')}
            <div class="empty-state">No prediction has been made in this browser session.</div>
        `;
        return;
    }

    const actions = `
        <button class="btn btn-primary" onclick="generateReport()">Generate Report</button>
        <button class="btn btn-secondary" onclick="navigateTo('predict')">New Screening</button>
    `;
    page.innerHTML = `
        ${pageHeader('Prediction result', 'Patient Prediction Result', 'Doctor-friendly summary with risk, severity, recommendation, and fairness note.', actions)}
        ${stepper(3)}
        <div id="result-content">${renderPredictionResult(state.lastResult)}</div>
        <div class="chart-grid" style="margin-top:1rem;">${chartCard('probabilityChart', 'Model confidence / probabilities')}</div>
    `;
    renderProbabilityChart(state.lastResult.probabilities || {});
}

function renderPredictionResult(result) {
    const riskBand = result.risk_band || getRiskBandFromSeverity(result.predicted_severity);
    const riskClass = riskBand.toLowerCase();
    const riskPercent = Math.round(Number(result.risk_percentage ?? result.risk_score * 100) || 0);
    const payload = state.lastPayload || result.input_data || {};

    return `
        <article class="report-card">
            <div class="result-hero">
                <div class="card" style="box-shadow:none;">
                    <div class="risk-ring" style="background:conic-gradient(var(--${riskClass === 'high' ? 'high' : riskClass === 'medium' ? 'medium' : 'low'}) ${riskPercent * 3.6}deg, #edf1ea 0deg);">
                        <div class="risk-ring-inner">
                            <div>
                                <div class="risk-number">${riskPercent}%</div>
                                <div class="section-note">Risk score</div>
                            </div>
                        </div>
                    </div>
                </div>
                <div>
                    <div class="button-row">
                        ${riskBadge(result.predicted_severity, riskBand)}
                        ${riskBadge(`${riskBand} risk`, riskBand)}
                        <span class="badge ${result.fairness_flag ? 'badge-medium' : 'badge-low'}">${result.fairness_flag ? 'Fairness flag shown' : 'No fairness flag'}</span>
                    </div>
                    <h2 style="margin:1rem 0 0; font-size:clamp(1.6rem,3vw,2.2rem);">${escapeHtml(result.patient_name || 'Unnamed patient')}</h2>
                    <p class="section-note">Patient ID: ${escapeHtml(result.patient_id || 'Not provided')} | Visit: ${escapeHtml(result.visit_date || '-')} ${escapeHtml(result.visit_time || '')}</p>
                    <div class="stat-grid">
                        ${statCard('PHQ-9 total', `${result.phq9_total}/27`)}
                        ${statCard('Predicted severity', result.predicted_severity)}
                        ${statCard('Risk band', riskBand)}
                    </div>
                    <div class="recommendation-box" style="margin-top:1rem;"><strong>Recommendation:</strong> ${escapeHtml(result.recommendation)}</div>
                    <div class="error-state" style="margin-top:0.75rem;"><strong>Warning:</strong> ${escapeHtml(result.warning)}</div>
                    <div class="success" style="margin-top:0.75rem;"><strong>Doctor-friendly explanation:</strong> ${escapeHtml(result.explanation || explanationForResult(result))}</div>
                    ${result.mongo_note ? `<div class="empty-state" style="margin-top:0.75rem;">${escapeHtml(result.mongo_note)}</div>` : ''}
                </div>
            </div>
            <h3 class="section-title" style="margin-top:1.25rem;">PHQ-9 answer breakdown</h3>
            ${renderPhqBreakdown(payload)}
        </article>
    `;
}

function renderPhqBreakdown(payload) {
    const rows = PHQ_QUESTIONS.map(([code, question]) => {
        const value = Number(payload?.[code] ?? 0);
        const label = PHQ_OPTIONS.find(([score]) => score === value)?.[1] || 'Unknown';
        return `<tr><td>${code}</td><td>${escapeHtml(question)}</td><td>${value}</td><td>${label}</td></tr>`;
    }).join('');
    return `<div class="table-wrap"><table><thead><tr><th>Code</th><th>Question</th><th>Score</th><th>Meaning</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function renderFairnessPage(page) {
    page.innerHTML = `
        ${pageHeader('Fairness analysis', 'Fairness', 'Compare risk and model performance by age, gender, race, and income group in the NHANES test data.')}
        <div class="tabs no-print">
            ${fairnessTabButton('AGE_GROUP', 'By Age Group')}
            ${fairnessTabButton('RIAGENDR', 'By Gender')}
            ${fairnessTabButton('RIDRETH1', 'By Race')}
            ${fairnessTabButton('INDHHIN2', 'By Income')}
            ${fairnessTabButton('CHECK', 'Check My Patient')}
        </div>
        <div id="fairness-content" class="main-content-card"></div>
    `;
    loadFairnessReport();
}

function fairnessTabButton(tab, label) {
    return `<button class="tab-btn ${state.fairnessTab === tab ? 'active' : ''}" onclick="renderFairnessTab('${tab}')">${label}</button>`;
}

function renderFairnessTab(type) {
    state.fairnessTab = type;
    renderPage();
}

async function loadFairnessReport() {
    const container = document.getElementById('fairness-content');
    if (!container) return;

    if (state.fairnessTab === 'CHECK') {
        renderCheckMyPatient(container);
        return;
    }

    showLoading(container, 'Loading fairness report...');
    try {
        const [report, summary] = await Promise.all([
            apiCall('/fairness-report').catch(() => ({ records: [] })),
            loadFairnessSummary().catch(() => ({}))
        ]);
        state.fairnessReport = report.records || [];
        state.fairnessSummary = summary;
        container.className = 'card';
        container.innerHTML = renderFairnessRows(state.fairnessTab, state.fairnessReport, summary);
    } catch (error) {
        showError(container, `Fairness page failed: ${error.message}`);
    }
}

async function loadFairnessSummary() {
    return apiCall('/fairness-summary');
}

function renderFairnessRows(type, records, summary = {}) {
    const rows = records.filter(item => item.group_column === type);
    if (!rows.length) {
        return '<div class="empty-state">No fairness data available. Run python scripts/bootstrap_ml.py or python -m mentalhealthiq.fairness.</div>';
    }

    const titles = {
        AGE_GROUP: 'Depression Risk by Age Group',
        RIAGENDR: 'Depression Risk by Gender',
        RIDRETH1: 'Depression Risk by Race',
        INDHHIN2: 'Depression Risk by Income'
    };
    const sorted = [...rows].sort((a, b) => riskPercentage(b) - riskPercentage(a));
    const highest = sorted[0];
    const lowest = sorted[sorted.length - 1];

    return `
        <h2 class="section-title">${titles[type] || 'Depression risk by group'}</h2>
        <p class="section-note">How risk levels differ across groups in the NHANES dataset.</p>
        ${sorted.map(row => {
            const percentage = Math.round(riskPercentage(row));
            const band = percentage >= 30 ? 'High' : percentage >= 20 ? 'Medium' : 'Low';
            return `
                <div class="fairness-row">
                    <strong>${escapeHtml(formatGroupValue(row.group_column, row.group_value))}</strong>
                    <div>
                        <div class="progress-track"><div class="progress-fill ${band.toLowerCase()}" style="width:${percentage}%"></div></div>
                        <small class="section-note">Accuracy ${roundPercent(row.accuracy)} | F1 ${roundPercent(row.f1)}</small>
                    </div>
                    <span class="badge badge-${band.toLowerCase()}">${percentage}% ${band}</span>
                </div>
            `;
        }).join('')}
        <div class="summary-pair">
            <div class="soft-card"><div class="stat-label">Highest risk group</div><div class="stat-value" style="color:var(--high);">${escapeHtml(formatGroupValue(highest.group_column, highest.group_value))}</div></div>
            <div class="soft-card"><div class="stat-label">Lowest risk group</div><div class="stat-value" style="color:var(--low);">${escapeHtml(formatGroupValue(lowest.group_column, lowest.group_value))}</div></div>
        </div>
        <div class="${summary.bias_detected ? 'error-state' : 'success'}" style="margin-top:1rem;">
            <strong>${summary.bias_detected ? 'Bias detected' : 'No major bias detected'}:</strong> ${escapeHtml(summary.fairness_warning || 'Fairness summary is not available.')}
        </div>
    `;
}

function renderCheckMyPatient(container) {
    container.className = 'card';
    container.innerHTML = `
        <h2 class="section-title">Check My Patient</h2>
        <p class="section-note">Use the latest prediction or enter patient groups manually. This gives a doctor-friendly fairness note.</p>
        <div class="form-grid">
            ${inputField('Age', 'fairAge', 'number', state.lastPayload?.RIDAGEYR || '35', 'min="0" max="120"')}
            ${selectField('Gender', 'fairGender', [['1', 'Male'], ['2', 'Female']], state.lastPayload?.RIAGENDR || '2')}
            ${selectField('Race / Ethnicity', 'fairRace', [['1', 'Mexican American'], ['2', 'Other Hispanic'], ['3', 'Non-Hispanic White'], ['4', 'Non-Hispanic Black'], ['5', 'Other Race']], state.lastPayload?.RIDRETH1 || '3')}
            ${selectField('Income', 'fairIncome', [['1', 'Low income'], ['2', 'Lower middle'], ['3', 'Middle'], ['4', 'Upper middle'], ['5', 'High income']], state.lastPayload?.INDHHIN2 || '3')}
            ${selectField('Risk Band', 'fairRiskBand', [['Low', 'Low'], ['Medium', 'Medium'], ['High', 'High']], state.lastResult?.risk_band || 'Low')}
            <div class="form-group"><label>&nbsp;</label><button class="btn btn-primary" onclick="checkMyPatientFairness()">Check Patient</button></div>
        </div>
        <div id="patient-fairness-result" style="margin-top:1rem;"></div>
    `;
}

async function checkMyPatientFairness() {
    const target = document.getElementById('patient-fairness-result');
    showLoading(target, 'Checking fairness summary...');
    const summary = await loadFairnessSummary().catch(() => ({}));
    const ageGroup = deriveAgeGroup(document.getElementById('fairAge')?.value);
    const riskBand = document.getElementById('fairRiskBand')?.value || 'Low';
    const highestAge = summary.highest_risk_age_group;
    const flag = Boolean(summary.bias_detected && (riskBand === 'High' || String(ageGroup) === String(highestAge)));
    target.className = flag ? 'error-state' : 'success';
    target.innerHTML = `
        <strong>Patient group:</strong> ${escapeHtml(ageGroup)}.<br>
        <strong>Risk band:</strong> ${escapeHtml(riskBand)}.<br>
        ${flag ? 'Fairness warning exists. Review group-level risk and model performance before making decisions.' : 'No strong fairness warning for this patient from the current report.'}
    `;
}

function renderHistoryPage(page) {
    page.innerHTML = `
        ${pageHeader('Saved predictions', 'Patient History', 'Search and review saved patient screenings. MongoDB must be configured for saved history.')}
        <div class="card">
            <div class="filter-bar">
                ${patientSearchField('Patient', 'historyPatientSearch', 'historyPatientsList')}
                ${inputField('Search Patient Name', 'historyName', 'text')}
                ${selectField('Severity', 'historySeverity', [['', 'All'], ...SEVERITIES.map(item => [item, item])])}
                ${selectField('Risk Band', 'historyRisk', [['', 'All'], ['Low', 'Low'], ['Medium', 'Medium'], ['High', 'High']])}
                ${inputField('From Date', 'historyFrom', 'date')}
                ${inputField('To Date', 'historyTo', 'date')}
            </div>
            <div id="historyPatientStatus" style="margin-top:1rem;"></div>
            <div class="button-row">
                <button class="btn btn-primary" onclick="loadHistory()">Load History</button>
                <button class="btn btn-secondary" onclick="searchPatientHistory()">Apply Filters</button>
            </div>
        </div>
        <div id="history-content" style="margin-top:1rem;"></div>
    `;
    loadHistory();
}

async function loadHistory() {
    showLoading('history-content', 'Loading patient history...');
    try {
        const response = await apiCall('/predictions?limit=500');
        state.history = normalizePredictionsResponse(response);
        await populatePatientSearchDropdown('historyPatientSearch', 'historyPatientsList', 'historyPatientStatus', state.history);
        await searchPatientHistory();
    } catch (error) {
        showError('historyPatientStatus', 'Could not load saved patients. Make sure API and MongoDB are running.');
        showError('history-content', `History failed: ${error.message}`);
    }
}

async function searchPatientHistory() {
    const target = document.getElementById('history-content');
    if (!target) return;

    const selectedPatientId = getSelectedPatientId('historyPatientSearch');
    const name = valueOf('historyName').toLowerCase();
    const severity = valueOf('historySeverity');
    const risk = valueOf('historyRisk');
    const from = valueOf('historyFrom');
    const to = valueOf('historyTo');
    let records = state.history;

    if (selectedPatientId) {
        showLoading(target, 'Loading selected patient history...');
        try {
            const response = await apiCall(`/patients/${encodeURIComponent(selectedPatientId)}/history`);
            records = normalizePredictionsResponse(response);
        } catch (error) {
            showError(target, `Patient history failed: ${error.message}`);
            return;
        }
    }

    const filtered = records.filter(item => {
        const itemName = getPatientName(item).toLowerCase();
        const itemSeverity = item.predicted_severity || item.severity || '';
        const itemRisk = item.risk_band || getRiskBandFromSeverity(itemSeverity);
        const itemDate = item.visit_date || String(item.timestamp || '').slice(0, 10);
        return (!name || itemName.includes(name))
            && (!severity || itemSeverity === severity)
            && (!risk || itemRisk === risk)
            && (!from || itemDate >= from)
            && (!to || itemDate <= to);
    });
    state.filteredHistory = filtered;

    target.className = 'card';
    target.innerHTML = renderHistoryRecords(filtered);
}

function renderHistoryRecords(records) {
    state.filteredHistory = records || [];
    if (!records || !records.length) {
        return '<div class="empty-state">No predictions saved yet. Submit a patient form to populate dashboard and history.</div>';
    }

    const rows = records.map((item, index) => historyRow(item, index)).join('');
    const cards = records.map((item, index) => historyMobileCard(item, index)).join('');
    return `
        <div class="table-wrap history-table">
            <table>
                <thead><tr><th>Patient</th><th>Date / Time</th><th>PHQ-9</th><th>Severity</th><th>Risk</th><th>Actions</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
        <div class="history-cards">${cards}</div>
    `;
}

function historyRow(item, index) {
    const severity = item.predicted_severity || item.severity || 'Unknown';
    const riskBand = item.risk_band || getRiskBandFromSeverity(severity);
    const patientId = item.patient_id || item.input_data?.patient_id || '';
    return `
        <tr>
            <td><strong>${escapeHtml(item.patient_name || item.input_data?.patient_name || 'Unnamed')}</strong><br><small>${escapeHtml(patientId || 'No ID')}</small></td>
            <td>${escapeHtml(item.visit_date || String(item.timestamp || '').slice(0, 10))}<br><small>${escapeHtml(item.visit_time || '')}</small></td>
            <td>${item.phq9_total ?? '-'}</td>
            <td>${riskBadge(severity, riskBand)}</td>
            <td>${percent(item.risk_score || 0)}<br><small>${riskBand}</small></td>
            <td><div class="button-row"><button class="btn btn-secondary" onclick="viewHistoryDetailByIndex(${index})">View</button><button class="btn btn-secondary" onclick="reportFromHistoryByIndex(${index})">Report</button></div></td>
        </tr>
    `;
}

function historyMobileCard(item, index) {
    const severity = item.predicted_severity || item.severity || 'Unknown';
    const riskBand = item.risk_band || getRiskBandFromSeverity(severity);
    return `
        <div class="history-card-item">
            <div><strong>${escapeHtml(item.patient_name || item.input_data?.patient_name || 'Unnamed')}</strong><br><small>${escapeHtml(item.patient_id || item.input_data?.patient_id || 'No ID')}</small></div>
            <div>${riskBadge(severity, riskBand)} ${riskBadge(`${percent(item.risk_score || 0)} risk`, riskBand)}</div>
            <div>PHQ-9: ${item.phq9_total ?? '-'} | ${escapeHtml(item.visit_date || String(item.timestamp || '').slice(0, 10))}</div>
            <div class="button-row"><button class="btn btn-secondary" onclick="viewHistoryDetailByIndex(${index})">View details</button><button class="btn btn-secondary" onclick="reportFromHistoryByIndex(${index})">Report</button></div>
        </div>
    `;
}

function viewHistoryDetailByIndex(index) {
    const item = state.filteredHistory[index];
    if (item) viewHistoryDetail(item);
}

function reportFromHistoryByIndex(index) {
    const item = state.filteredHistory[index];
    if (item) reportFromHistory(item);
}

function viewHistoryDetail(item) {
    state.lastResult = normalizeResult(item, item.input_data || {});
    state.lastPayload = item.input_data || {};
    writeStorage('mhiq_last_result', state.lastResult);
    writeStorage('mhiq_last_payload', state.lastPayload);
    navigateTo('result');
}

function reportFromHistory(item) {
    viewHistoryDetail(item);
    navigateTo('reports');
}

function renderComparisonPage(page) {
    page.innerHTML = `
        ${pageHeader('Patient trend', 'Patient Comparison', 'Compare the current/latest saved visit with the previous saved visit for a patient.')}
        <div class="card">
            <div class="form-grid">
                ${patientSearchField('Patient', 'comparisonPatientSearch', 'comparisonPatientsList')}
                <div class="form-group"><label>&nbsp;</label><button id="comparePatientBtn" class="btn btn-primary" onclick="loadPatientComparison()" disabled>Compare Patient Visits</button></div>
            </div>
            <div id="comparisonStatus" style="margin-top:1rem;"></div>
        </div>
        <div id="comparison-content" style="margin-top:1rem;"></div>
    `;
    showEmpty('comparison-content', 'Select a patient to compare previous and current visits.');
    loadComparisonPatients();
}

async function loadComparisonPatients() {
    const button = document.getElementById('comparePatientBtn');
    if (!button) return;

    button.disabled = true;

    try {
        const patients = await populatePatientSearchDropdown(
            'comparisonPatientSearch',
            'comparisonPatientsList',
            'comparisonStatus'
        );
        state.comparisonPatients = patients;

        const lastPatientId = state.lastPayload?.patient_id || state.lastResult?.patient_id || '';
        const input = document.getElementById('comparisonPatientSearch');
        if (input && lastPatientId && patients.some(patient => patient.patient_id === lastPatientId)) {
            input.value = lastPatientId;
        }

        if (!patients.length) {
            const input = document.getElementById('comparisonPatientSearch');
            if (input?.dataset.patientLoadFailed === 'true') {
                showEmpty('comparison-content', 'Saved patients are unavailable right now.');
            } else {
                showEmpty('comparison-content', 'No saved patients found. Use Predict & Save first.');
            }
            return;
        }

        button.disabled = false;
    } catch (error) {
        state.comparisonPatients = [];
        showEmpty('comparison-content', 'Saved patients are unavailable right now.');
    }
}

async function loadPatientComparison() {
    const patientId = getSelectedPatientId('comparisonPatientSearch');
    if (!patientId) {
        showError('comparison-content', 'Select a patient first.');
        return;
    }
    showLoading('comparison-content', 'Loading patient comparison...');
    try {
        const data = await apiCall(`/patients/${encodeURIComponent(patientId)}/comparison`);
        renderComparisonResult(data);
    } catch (error) {
        showError('comparison-content', `Comparison failed: ${error.message}`);
    }
}

const comparePatient = loadPatientComparison;

function renderComparisonResult(data) {
    const target = document.getElementById('comparison-content');
    if (!data.has_comparison) {
        showEmpty(target, data.message || data.summary || 'At least two saved visits are needed for comparison.');
        return;
    }

    const current = data.current || {};
    const previous = data.previous || {};
    const riskDelta = Math.round((Number(current.risk_score || 0) - Number(previous.risk_score || 0)) * 100);
    const status = String(data.status || '').toLowerCase();
    const change = data.change ?? data.phq9_change ?? 0;
    const statusClass = status === 'improved' ? 'success' : status === 'worsened' ? 'error-state' : 'empty-state';
    target.className = 'card';
    target.innerHTML = `
        <div class="stat-grid">
            ${statCard('Previous PHQ-9', previous.phq9_total ?? '-')}
            ${statCard('Current PHQ-9', current.phq9_total ?? '-')}
            ${statCard('PHQ-9 difference', change > 0 ? `+${change}` : change)}
            ${statCard('Risk difference', riskDelta > 0 ? `+${riskDelta}%` : `${riskDelta}%`)}
            ${statCard('Previous severity', previous.predicted_severity || previous.severity || '-')}
            ${statCard('Current severity', current.predicted_severity || current.severity || '-')}
        </div>
        <div class="${statusClass}" style="margin-top:1rem;"><strong>${escapeHtml(data.status || 'Comparison')}:</strong> ${escapeHtml(data.message || data.summary || '')}</div>
        <div class="chart-grid" style="margin-top:1rem;">${chartCard('historyTrendChart', 'Trend over time')}</div>
    `;
    renderHistoryTrendChart(data.trend || []);
}

function renderReportsPage(page) {
    const actions = `
        <button class="btn btn-primary" onclick="printReport()">Print Report</button>
        <button class="btn btn-secondary" onclick="navigateTo('history')">Back to History</button>
    `;
    page.innerHTML = `
        ${pageHeader('Printable report', 'Patient Report', 'Print or save the current prediction report from the browser.', actions)}
        <div class="card no-print" style="margin-bottom:1rem;">
            <div class="form-grid">
                ${patientSearchField('Patient', 'reportPatientSearch', 'reportPatientsList')}
                <div class="form-group"><label>&nbsp;</label><button id="loadReportPatientBtn" class="btn btn-primary" onclick="loadReportForSelectedPatient()" disabled>Load Latest Report</button></div>
            </div>
            <div id="reportPatientStatus" style="margin-top:1rem;"></div>
        </div>
        <div id="report-content">${state.lastResult ? renderPrintableReport() : '<div class="empty-state">No report available. Submit a PHQ-9 form or open a saved history record first.</div>'}</div>
    `;
    loadReportPatients();
}

function generateReport() {
    navigateTo('reports');
}

async function loadReportPatients() {
    const button = document.getElementById('loadReportPatientBtn');
    if (!button) return;

    button.disabled = true;
    const patients = await populatePatientSearchDropdown('reportPatientSearch', 'reportPatientsList', 'reportPatientStatus');
    state.reportPatients = patients;

    const lastPatientId = state.lastPayload?.patient_id || state.lastResult?.patient_id || '';
    const input = document.getElementById('reportPatientSearch');
    if (input && lastPatientId && patients.some(patient => patient.patient_id === lastPatientId)) {
        input.value = lastPatientId;
    }

    button.disabled = patients.length === 0;
}

async function loadReportForSelectedPatient() {
    const patientId = getSelectedPatientId('reportPatientSearch');
    const target = document.getElementById('report-content');
    if (!patientId) {
        showError('reportPatientStatus', 'Select a patient first.');
        return;
    }

    showLoading(target, 'Loading latest patient report...');
    try {
        const response = await apiCall(`/patients/${encodeURIComponent(patientId)}/history`);
        const records = normalizePredictionsResponse(response);
        if (!records.length) {
            showEmpty(target, 'No saved visits found for this patient.');
            return;
        }

        const latest = [...records].sort((a, b) => String(getVisitValue(b)).localeCompare(String(getVisitValue(a))))[0];
        state.lastPayload = latest.input_data || {};
        state.lastResult = normalizeResult(latest, state.lastPayload);
        writeStorage('mhiq_last_payload', state.lastPayload);
        writeStorage('mhiq_last_result', state.lastResult);
        target.innerHTML = renderPrintableReport();
    } catch (error) {
        showError(target, `Report failed: ${error.message}`);
    }
}

function printReport() {
    window.print();
}

function renderPrintableReport() {
    const result = state.lastResult;
    const payload = state.lastPayload || result.input_data || {};
    return `
        <article class="report-card report-page">
            <h1>MentalHealthIQ Patient Report</h1>
            <p class="section-note">PHQ-9 Depression Severity Screening</p>
            <div class="report-meta" style="margin:1rem 0;">
                ${statCard('Patient name', result.patient_name || payload.patient_name || 'Unnamed')}
                ${statCard('Patient ID', result.patient_id || payload.patient_id || 'No ID')}
                ${statCard('Visit date', result.visit_date || payload.visit_date || '-')}
                ${statCard('Visit time', result.visit_time || payload.visit_time || '-')}
            </div>
            ${renderPredictionResult(result)}
            <div class="card" style="box-shadow:none; margin-top:1rem;">
                <h2 class="section-title">Doctor notes</h2>
                <p>${escapeHtml(result.doctor_notes || payload.doctor_notes || 'No notes added.')}</p>
            </div>
        </article>
    `;
}

function renderMetricsPage(page) {
    page.innerHTML = `
        ${pageHeader('Model performance', 'Metrics', 'Model accuracy, precision, recall, F1 score, confusion matrix, and classification report.')}
        <div id="metrics-content" class="main-content-card"></div>
    `;
    loadMetrics();
}

async function loadMetrics() {
    showLoading('metrics-content', 'Loading model metrics...');
    try {
        const data = await apiCall('/metrics');
        const metrics = data.metrics || {};
        const target = document.getElementById('metrics-content');
        target.className = '';
        target.innerHTML = `
            <div class="metrics-grid">
                ${statCard('Best model', metrics.best_model_name || 'N/A')}
                ${statCard('Accuracy', decimal(metrics.accuracy))}
                ${statCard('Precision', decimal(metrics.precision))}
                ${statCard('Recall', decimal(metrics.recall))}
                ${statCard('F1 score', decimal(metrics.f1))}
            </div>
            <div class="card" style="margin-top:1rem;">
                <h2 class="section-title">Confusion matrix</h2>
                ${renderMatrix(metrics.confusion_matrix || data.confusion_matrix)}
            </div>
            <div class="card" style="margin-top:1rem;">
                <h2 class="section-title">Classification report</h2>
                ${renderClassificationReport(metrics.classification_report)}
            </div>
        `;
    } catch (error) {
        showError('metrics-content', `Model metrics not generated yet. Run python scripts/bootstrap_ml.py. Details: ${error.message}`);
    }
}

const loadMetricsData = loadMetrics;

function renderHealthPage(page) {
    page.innerHTML = `
        ${pageHeader('System readiness', 'Health', 'Clear readiness status for API, model, preprocessor, fairness report, and MongoDB.')}
        <div id="health-content" class="main-content-card"></div>
    `;
    loadHealth();
}

async function loadHealth() {
    showLoading('health-content', 'Checking system health...');
    try {
        const health = await apiCall('/health');
        const target = document.getElementById('health-content');
        target.className = '';
        target.innerHTML = `
            <div class="stat-grid">
                ${healthCard('API', health.api)}
                ${healthCard('Model', health.model)}
                ${healthCard('Preprocessor', health.preprocessor)}
                ${healthCard('Fairness Report', health.fairness_report)}
                ${healthCard('MongoDB', health.mongo)}
            </div>
            ${(health.model !== 'ready' || health.preprocessor !== 'ready' || health.fairness_report !== 'ready') ? '<div class="empty-state" style="margin-top:1rem;">Missing artifacts? Run <strong>python scripts/bootstrap_ml.py</strong>.</div>' : ''}
            ${mongoHealthHelp(health.mongo)}
        `;
    } catch (error) {
        showError('health-content', `Health check failed: ${error.message}`);
    }
}

function safeRenderChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return;
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }
    charts[canvasId] = new Chart(canvas, config);
}

function destroyCharts() {
    Object.keys(charts).forEach(id => {
        charts[id].destroy();
        delete charts[id];
    });
}

function chartCard(id, title) {
    return `<div class="chart-card"><h3 class="section-title">${title}</h3><div class="chart-box"><canvas id="${id}"></canvas></div></div>`;
}

function renderSeverityChart(counts) {
    const labels = Object.keys(counts);
    const values = Object.values(counts);
    safeRenderChart('severityChart', doughnutConfig(labels, values, ['#178f6d', '#7fc8a9', '#f2a51a', '#ef7c55', '#de3f45']));
}

function renderRiskBandChart(counts) {
    const labels = Object.keys(counts);
    const values = Object.values(counts);
    safeRenderChart('riskBandChart', doughnutConfig(labels, values, ['#178f6d', '#f2a51a', '#de3f45']));
}

function renderTimeCharts(records) {
    const byDate = {};
    records.forEach(item => {
        const date = item.visit_date || String(item.timestamp || '').slice(0, 10);
        if (!date) return;
        byDate[date] ||= { count: 0, phqTotal: 0 };
        byDate[date].count += 1;
        byDate[date].phqTotal += Number(item.phq9_total || 0);
    });
    const labels = Object.keys(byDate).sort();
    safeRenderChart('predictionsTimeChart', lineConfig(labels, labels.map(date => byDate[date].count), 'Predictions'));
    safeRenderChart('phqAverageChart', lineConfig(labels, labels.map(date => byDate[date].count ? Number((byDate[date].phqTotal / byDate[date].count).toFixed(2)) : 0), 'Average PHQ-9'));
}

function renderFairnessDashboardCharts(records) {
    const gender = records.filter(item => item.group_column === 'RIAGENDR');
    const age = records.filter(item => item.group_column === 'AGE_GROUP');
    safeRenderChart('genderRiskChart', barConfig(gender.map(item => formatGroupValue(item.group_column, item.group_value)), gender.map(item => Math.round(riskPercentage(item))), 'Risk %'));
    safeRenderChart('ageRiskChart', barConfig(age.map(item => formatGroupValue(item.group_column, item.group_value)), age.map(item => Math.round(riskPercentage(item))), 'Risk %'));
}

function renderProbabilityChart(probabilities) {
    const labels = Object.keys(probabilities);
    const values = Object.values(probabilities).map(value => Math.round(Number(value) * 100));
    safeRenderChart('probabilityChart', barConfig(labels, values, 'Probability %'));
}

function renderHistoryTrendChart(records) {
    const labels = records.map(item => item.visit_date || String(item.timestamp || '').slice(0, 10));
    const phq = records.map(item => Number(item.phq9_total || 0));
    const risk = records.map(item => Math.round(Number(item.risk_score || 0) * 100));
    safeRenderChart('historyTrendChart', {
        type: 'line',
        data: {
            labels,
            datasets: [
                { label: 'PHQ-9 total', data: phq, borderColor: '#147d67', backgroundColor: '#147d67', tension: 0.35 },
                { label: 'Risk %', data: risk, borderColor: '#de3f45', backgroundColor: '#de3f45', tension: 0.35 }
            ]
        },
        options: chartOptions()
    });
}

function doughnutConfig(labels, values, colors) {
    return {
        type: 'doughnut',
        data: { labels, datasets: [{ data: values, backgroundColor: colors }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } }
    };
}

function barConfig(labels, values, label) {
    return {
        type: 'bar',
        data: { labels, datasets: [{ label, data: values, backgroundColor: '#147d67', borderRadius: 8 }] },
        options: chartOptions()
    };
}

function lineConfig(labels, values, label) {
    return {
        type: 'line',
        data: { labels, datasets: [{ label, data: values, borderColor: '#147d67', backgroundColor: '#147d67', tension: 0.35 }] },
        options: chartOptions()
    };
}

function chartOptions() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: { y: { beginAtZero: true } }
    };
}

function statCard(label, value, help = '') {
    return `<div class="stat-card"><div class="stat-label">${escapeHtml(label)}</div><div class="stat-value">${escapeHtml(value ?? 'N/A')}</div>${help ? `<div class="stat-help">${escapeHtml(help)}</div>` : ''}</div>`;
}

function healthCard(label, value) {
    const normalized = String(value || 'unknown').toLowerCase();
    const badgeClass = normalized === 'ready' ? 'badge-low' : normalized === 'missing' || normalized === 'error' ? 'badge-high' : 'badge-medium';
    const displayValue = normalized === 'error' && label === 'MongoDB' ? 'connection error' : value || 'unknown';
    return `<div class="stat-card"><div class="stat-label">${escapeHtml(label)}</div><div class="stat-value"><span class="badge ${badgeClass}">${escapeHtml(displayValue)}</span></div></div>`;
}

function mongoHealthHelp(status) {
    const normalized = String(status || '').toLowerCase();
    if (normalized === 'ready') return '';
    if (normalized === 'not_configured') {
        return '<div class="error-state" style="margin-top:1rem;">Using local fallback failed. Check backend MongoDB config.</div>';
    }
    if (normalized !== 'error') return '';
    return `
        <div class="error-state" style="margin-top:1rem;">
            <strong>MongoDB is not running.</strong>
            <div style="margin-top:0.5rem;">Local MongoDB setup:</div>
            <ol style="margin:0.5rem 0 0 1.25rem;">
                <li>Start MongoDB locally or run: <strong>docker compose up -d mongodb mongo-express</strong></li>
                <li>Open MongoDB Compass: <strong>mongodb://localhost:27017</strong></li>
                <li>Submit a prediction using Predict and Save.</li>
                <li>Check database: <strong>mentalhealthiq -> predictions</strong></li>
            </ol>
        </div>
    `;
}

function riskBadge(text, riskBand) {
    const riskClass = String(riskBand || 'Low').toLowerCase();
    return `<span class="badge badge-${riskClass}">${escapeHtml(text)}</span>`;
}

function renderMatrix(matrix) {
    if (!Array.isArray(matrix) || !matrix.length) {
        return '<div class="empty-state">Confusion matrix not available.</div>';
    }
    return `<div class="table-wrap"><table><tbody>${matrix.map((row, index) => `<tr><th>Actual ${index}</th>${row.map(cell => `<td>${cell}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function renderClassificationReport(report) {
    if (!report || typeof report !== 'object') {
        return '<div class="empty-state">Classification report not available.</div>';
    }
    const rows = Object.entries(report)
        .filter(([, value]) => value && typeof value === 'object')
        .map(([label, value]) => `<tr><td>${escapeHtml(label)}</td><td>${decimal(value.precision)}</td><td>${decimal(value.recall)}</td><td>${decimal(value['f1-score'])}</td><td>${value.support ?? '-'}</td></tr>`)
        .join('');
    return `<div class="table-wrap"><table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

function recommendationForSeverity(severity) {
    if (severity === 'Minimal') return 'Continue routine wellness monitoring.';
    if (severity === 'Mild') return 'Schedule follow-up if symptoms continue or increase.';
    if (severity === 'Moderate') return 'Mental health follow-up is suggested.';
    return 'Immediate mental health follow-up is advised.';
}

function warningForSeverity(severity) {
    if (severity === 'Minimal' || severity === 'Mild') return 'Low warning: monitor symptoms and seek support if symptoms persist.';
    if (severity === 'Moderate') return 'Follow-up suggested: consider contacting a qualified mental health professional.';
    return 'Urgent follow-up advised: contact a qualified mental health professional promptly.';
}

function explanationForResult(result) {
    return `The PHQ-9 score is ${result.phq9_total}, which is in the ${result.predicted_severity} range. The model estimates ${String(result.risk_band).toLowerCase()} risk from PHQ-9 answers and demographic fields.`;
}

function formatGroupValue(column, value) {
    const maps = {
        RIAGENDR: { 1: 'Male', 2: 'Female' },
        RIDRETH1: { 1: 'Mexican American', 2: 'Other Hispanic', 3: 'Non-Hispanic White', 4: 'Non-Hispanic Black', 5: 'Other Race' },
        INDHHIN2: { 1: 'Low income', 2: 'Lower middle', 3: 'Middle', 4: 'Upper middle', 5: 'High income', 77: 'Refused', 99: 'Unknown' }
    };
    return maps[column]?.[value] || value;
}

function riskPercentage(row) {
    return Number(row.risk_percentage ?? (Number(row.selection_rate || 0) * 100));
}

function valueOf(id) {
    return String(document.getElementById(id)?.value || '').trim();
}

function setInputValue(id, value) {
    const element = document.getElementById(id);
    if (!element || value === undefined || value === null || value === '') return;
    element.value = String(value);
}

function today() {
    return new Date().toISOString().slice(0, 10);
}

function currentTime() {
    return new Date().toTimeString().slice(0, 5);
}

function formatDateTime(value) {
    if (!value) return 'N/A';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function percent(value) {
    return `${Math.round(Number(value || 0) * 100)}%`;
}

function roundPercent(value) {
    return `${Math.round(Number(value || 0) * 100)}%`;
}

function decimal(value) {
    return typeof value === 'number' ? value.toFixed(3) : 'N/A';
}

function average(values) {
    if (!values.length) return 0;
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

document.addEventListener('DOMContentLoaded', () => {
    setActiveNav(state.page);
    renderPage();
});
