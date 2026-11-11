// =========================================================================
// NIRMAAN FRONTEND CONTROLLER (Full Civic Defect Lifecycle)
// =========================================================================

const API_BASE = (window.location.port === "5000" || window.location.origin.includes(":5000"))
  ? window.location.origin
  : "http://127.0.0.1:5000";

const PREDICT_URL = `${API_BASE}/predict`;
const MATCH_URL = `${API_BASE}/api/match`;
const HEALTH_URL = `${API_BASE}/health`;

// State tracking
let currentActivePortal = "citizen";
let currentUserId = "usr_citizen_1";
let currentUserName = "Priya Malhotra";
let currentAIData = null;
let currentUploadedFile = null;
let selectedProblemId = null;
let allReportsCache = [];

// =========================================================================
// 1. AUTHENTICATION & PORTAL SWITCHER
// =========================================================================

function switchPortal(portal) {
  currentActivePortal = portal;

  // Toggle nav buttons
  document.querySelectorAll('.portal-btn').forEach(b => b.classList.remove('active'));
  const activeBtn = document.getElementById(
    portal === 'citizen' ? 'btnPortalCitizen' : (portal === 'contributor' ? 'btnPortalContrib' : 'btnPortalAdmin')
  );
  if (activeBtn) activeBtn.classList.add('active');

  // Toggle views
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  const viewElem = document.getElementById(
    portal === 'citizen' ? 'viewCitizen' : (portal === 'contributor' ? 'viewContributor' : 'viewAdmin')
  );
  if (viewElem) viewElem.classList.add('active');

  // Sync user selector to reasonable default for role if not matching
  const userSelect = document.getElementById('userSelector');
  if (userSelect) {
    if (portal === 'admin') {
      userSelect.value = 'usr_admin_1';
    } else if (portal === 'contributor' && userSelect.value === 'usr_citizen_1') {
      userSelect.value = 'usr_contractor_1';
    } else if (portal === 'citizen' && userSelect.value === 'usr_admin_1') {
      userSelect.value = 'usr_citizen_1';
    }
    currentUserId = userSelect.value;
  }

  // Load portal-specific data
  if (portal === 'citizen') {
    loadProblemBoard('all');
    loadMyReports();
    loadRewards();
    // Re-validate Leaflet map size after it becomes visible again
    setTimeout(() => {
      if (_leafletMaps['heroMap'])  _leafletMaps['heroMap'].invalidateSize();
      if (_leafletMaps['boardMap']) _leafletMaps['boardMap'].invalidateSize();
    }, 150);
  } else if (portal === 'contributor') {
    loadContributorTasks(currentContributorSubRole || 'contractor');
    loadContributorPayments(currentContributorSubRole || 'contractor');
  } else if (portal === 'admin') {
    loadAdminDashboard();
    loadAdminVerifications();
    loadAdminReportsTable();
  }

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function changeActiveUser(userId) {
  currentUserId = userId;
  const userSelect = document.getElementById('userSelector');
  if (userSelect) {
    const selectedOption = userSelect.options[userSelect.selectedIndex];
    currentUserName = selectedOption ? selectedOption.text.split('(')[0].trim() : "Citizen";
  }

  // Switch portal automatically if user selected an admin or contractor
  if (userId === 'usr_admin_1' && currentActivePortal !== 'admin') {
    switchPortal('admin');
  } else if (['usr_contractor_1', 'usr_engineer_1', 'usr_ngo_1', 'usr_csr_1'].includes(userId) && currentActivePortal !== 'contributor') {
    switchPortal('contributor');
  } else {
    // Refresh current view with new user
    if (currentActivePortal === 'citizen') {
      loadMyReports();
      loadRewards();
    } else if (currentActivePortal === 'contributor') {
      loadContributorTasks(currentContributorSubRole || 'contractor');
      loadContributorPayments(currentContributorSubRole || 'contractor');
    }
  }
}

function goToReportFlow() {
  if (currentActivePortal !== 'citizen') {
    switchPortal('citizen');
  }
  const reportSec = document.getElementById('report');
  if (reportSec) {
    reportSec.scrollIntoView({ behavior: 'smooth' });
  }
}

// =========================================================================
// 2. REAL NIRMAAN AI RECOGNITION (ResNet-50 + ResNet-18 Severity)
// =========================================================================

const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const uploadPreviewContainer = document.getElementById('uploadPreviewContainer');
const imagePreview = document.getElementById('imagePreview');
const uploadFileName = document.getElementById('uploadFileName');
const changePhotoBtn = document.getElementById('changePhotoBtn');

const aiIdle = document.getElementById('aiIdle');
const aiLoading = document.getElementById('aiLoading');
const aiResult = document.getElementById('aiResult');
const loadLine1 = document.getElementById('loadLine1');
const loadLine2 = document.getElementById('loadLine2');
const loadLine3 = document.getElementById('loadLine3');
const aiDeviceBadge = document.getElementById('aiDeviceBadge');

async function checkBackendHealth() {
  try {
    const res = await fetch(HEALTH_URL);
    if (res.ok) {
      const data = await res.json();
      if (aiDeviceBadge) {
        const dev = data.recognition_model?.device || "ONLINE";
        aiDeviceBadge.textContent = dev.toUpperCase().includes("CUDA") ? "GPU: ACTIVE" : "CPU: ACTIVE";
        aiDeviceBadge.style.color = "#6FC79A";
        aiDeviceBadge.style.background = "rgba(60,131,97,0.2)";
      }
    }
  } catch (err) {
    if (aiDeviceBadge) {
      aiDeviceBadge.textContent = "BACKEND DISCONNECTED";
      aiDeviceBadge.style.color = "#E08268";
      aiDeviceBadge.style.background = "rgba(192,80,58,0.2)";
    }
  }
}

function showPreview(fileOrUrl, name) {
  if (typeof fileOrUrl === "string") {
    imagePreview.src = fileOrUrl;
  } else {
    imagePreview.src = URL.createObjectURL(fileOrUrl);
  }
  uploadPlaceholder.style.display = 'none';
  uploadPreviewContainer.style.display = 'block';
  uploadFileName.textContent = name || "Uploaded Photo";
}

async function fetchAndRenderMatches(aiData) {
  const matchesContainer = document.getElementById('rMatches');
  if (!matchesContainer) return;

  matchesContainer.innerHTML = `
    <div class="match-loading">
      <div class="spinner" style="width:16px; height:16px; border-width:2px; display:inline-block; vertical-align:middle; margin:0 8px 0 0;"></div>
      Querying contributor database…
    </div>
  `;

  const locationInput = document.getElementById('locationInput');
  const locationText = (locationInput && locationInput.value.trim()) ? locationInput.value.trim() : "";

  const matchPayload = {
    category: aiData.category,
    severity: aiData.severity || "Medium",
    severity_score: aiData.severity_score !== undefined ? aiData.severity_score : 5.0,
    estimated_cost_min: (aiData.estimated_cost && aiData.estimated_cost.min) ? aiData.estimated_cost.min : 0,
    estimated_cost_max: (aiData.estimated_cost && aiData.estimated_cost.max) ? aiData.estimated_cost.max : 0,
    work_scope: aiData.work_scope || "",
    location_text: locationText
  };

  try {
    const res = await fetch(MATCH_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(matchPayload)
    });

    if (!res.ok) throw new Error(`Matching API error ${res.status}`);

    const data = await res.json();
    if (!data.success || !Array.isArray(data.matches) || data.matches.length === 0) {
      matchesContainer.innerHTML = `<div class="match-empty">No suitable contributors found for this location.</div>`;
      return;
    }

    matchesContainer.innerHTML = data.matches.slice(0, 3).map(m => {
      const distStr = (m.distance_km !== null && m.distance_km !== undefined) ? ` · ${m.distance_km}km away` : '';
      const primaryReason = (m.reasons && m.reasons.length > 0) ? m.reasons[0] : '';
      return `
        <div class="match-item">
          <div class="match-head">
            <div>
              <b>${m.name}</b>
              <span class="match-org">· ${m.organization}</span>
            </div>
            <span class="match-score">${m.match_score}% fit${distStr}</span>
          </div>
          ${primaryReason ? `<div class="match-reasons"><span class="match-reason">✓ ${primaryReason}</span></div>` : ''}
        </div>
      `;
    }).join('');
  } catch (err) {
    console.error("Matching Error:", err);
    matchesContainer.innerHTML = `<div class="match-error">Contributor ranking currently offline.</div>`;
  }
}

async function sendImageToAI(file, displayName) {
  currentUploadedFile = file;
  aiIdle.style.display = 'none';
  aiResult.classList.remove('show');
  aiLoading.classList.add('show');

  loadLine1.textContent = "Uploading image to Nirmaan AI API…";
  loadLine2.style.opacity = 0;
  loadLine3.style.opacity = 0;

  const t1 = setTimeout(() => {
    loadLine2.textContent = "Running ResNet-50 neural network…";
    loadLine2.style.opacity = 0.8;
  }, 350);

  const t2 = setTimeout(() => {
    loadLine3.textContent = "CategoryConditioned ResNet-18 Severity assessment…";
    loadLine3.style.opacity = 0.8;
  }, 700);

  try {
    const formData = new FormData();
    formData.append("image", file, displayName || "photo.jpg");

    const response = await fetch(PREDICT_URL, {
      method: "POST",
      body: formData
    });

    clearTimeout(t1);
    clearTimeout(t2);

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.error || `HTTP error ${response.status}`);
    }

    const data = await response.json();
    if (!data.success) throw new Error(data.error || "Inference failed");

    currentAIData = data;

    // Populate UI
    document.getElementById('rCategory').textContent = data.display_name || data.category;
    const rConf = document.getElementById('rConfidence');
    if (rConf) rConf.textContent = `${data.confidence_percentage} confidence`;

    const badge = document.getElementById('rSeverity');
    badge.textContent = data.severity_display || data.severity || "Medium";
    badge.className = 'badge ' + (data.severity_class || 'mid');

    document.getElementById('rReason').textContent = data.severity_reason || "Civic hazard detected by AI model";
    const costStr = (data.estimated_cost && data.estimated_cost.label) || data.estimated_repair_cost || "₹5,000 – ₹10,000";
    const rCostElem = document.getElementById('rCost');
    if (rCostElem) {
      rCostElem.textContent = costStr;
      if (data.work_scope) rCostElem.title = `Scope: ${data.work_scope}`;
    }

    const rModel = document.getElementById('rModelVerify');
    if (rModel) {
      const sevScore = data.severity_score !== undefined ? ` · ${data.severity_score}/10` : '';
      rModel.textContent = `✓ ResNet-50 (${data.category}) + ResNet-18 (${data.severity || 'Medium'}${sevScore}) + Cost Engine`;
    }

    // Call matching API
    fetchAndRenderMatches(data);

    aiLoading.classList.remove('show');
    aiResult.classList.add('show');

  } catch (error) {
    console.error("AI Error:", error);
    aiLoading.classList.remove('show');
    aiIdle.style.display = 'block';
    aiIdle.innerHTML = `<span style="color:#E08268; font-weight:600;">⚠ Inference Error:</span> ${error.message}`;
  }
}

// =========================================================================
// 3. CITIZEN PUBLISH REPORT TO DATABASE
// =========================================================================

async function submitReportToDatabase() {
  if (!currentAIData) {
    alert("Please upload and analyze a photo first.");
    return;
  }

  const publishBtn = document.getElementById('publishBtn');
  const oldText = publishBtn.textContent;
  publishBtn.disabled = true;
  publishBtn.textContent = "Publishing to live database…";

  const locInput = document.getElementById('locationInput');
  const locText = (locInput && locInput.value.trim()) ? locInput.value.trim() : "";

  try {
    const formData = new FormData();
    if (currentUploadedFile) {
      formData.append("image", currentUploadedFile);
    }
    formData.append("category", currentAIData.category || "potholes");
    formData.append("title", `${(currentAIData.display_name || currentAIData.category).capitalize()} defect, ${locText}`);
    formData.append("location_text", locText);
    formData.append("user_id", currentUserId);
    formData.append("ai_category", currentAIData.category);
    formData.append("ai_confidence", currentAIData.confidence || 0.95);
    formData.append("severity", currentAIData.severity || "Medium");
    formData.append("severity_score", currentAIData.severity_score !== undefined ? currentAIData.severity_score : 5.0);
    formData.append("severity_reason", currentAIData.severity_reason || "");

    const costMin = (currentAIData.estimated_cost && currentAIData.estimated_cost.min) ? currentAIData.estimated_cost.min : 6000;
    const costMax = (currentAIData.estimated_cost && currentAIData.estimated_cost.max) ? currentAIData.estimated_cost.max : 10000;
    formData.append("estimated_cost_min", costMin);
    formData.append("estimated_cost_max", costMax);
    formData.append("goal_amount", Math.round((costMin + costMax) / 2));
    formData.append("work_scope", currentAIData.work_scope || "Repair road and infrastructure.");

    const res = await fetch(`${API_BASE}/api/reports`, {
      method: "POST",
      body: formData
    });

    let result;
    const contentType = res.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      result = await res.json();
    } else {
      const errText = await res.text();
      throw new Error(`Server error (${res.status}): ${errText.slice(0, 100)}`);
    }
    if (!result.success) throw new Error(result.error || "Failed to create report");

    // Success feedback
    publishBtn.textContent = "✓ Published! Opening campaign…";
    setTimeout(() => {
      publishBtn.disabled = false;
      publishBtn.textContent = oldText;

      // Refresh boards and scroll to problem board
      loadProblemBoard('all', () => {
        selectProblem(result.report_id);
      });
      loadMyReports();
      document.getElementById('board').scrollIntoView({ behavior: 'smooth' });
    }, 800);

  } catch (err) {
    alert("Error publishing report: " + err.message);
    publishBtn.disabled = false;
    publishBtn.textContent = oldText;
  }
}

String.prototype.capitalize = function() {
  return this.charAt(0).toUpperCase() + this.slice(1);
};

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// =========================================================================
// 4. LIVE PROBLEM BOARD & FUNDING CAMPAIGNS
// =========================================================================

function getStatusBadgeHtml(status) {
  const clean = (status || 'REPORT').toUpperCase();
  const labelMap = {
    'REPORT': 'Reported',
    'FUNDING': 'In Funding',
    'FUNDED': 'Fully Funded',
    'ASSIGNED': 'Assigned',
    'IN_PROGRESS': 'In Progress',
    'SUBMITTED_FOR_VERIFICATION': 'Pending Verification',
    'VERIFIED': 'Verified',
    'REWORK': 'Needs Rework',
    'COMPLETED': 'Completed'
  };
  return `<span class="st-badge st-${clean}">${labelMap[clean] || clean}</span>`;
}

function getCategoryIcon(cat) {
  const map = {
    'potholes': '🕳',
    'streetlights': '💡',
    'waterlogging': '🚰',
    'garbage': '🗑'
  };
  return map[cat] || '🏗';
}

function getCategoryColor(cat) {
  const map = {
    'potholes': 'var(--terracotta)',
    'streetlights': 'var(--marigold)',
    'waterlogging': 'var(--ink-soft)',
    'garbage': 'var(--green)'
  };
  return map[cat] || 'var(--ink)';
}

async function loadProblemBoard(filterCategory = 'all', callback = null) {
  const container = document.getElementById('problemCardsContainer');
  if (!container) return;

  try {
    let url = `${API_BASE}/api/reports`;
    if (filterCategory && filterCategory !== 'all') {
      url += `?category=${encodeURIComponent(filterCategory)}`;
    }

    const res = await fetch(url);
    const data = await res.json();

    const heroCount = document.getElementById('heroReportCount');
    const heroFunded = document.getElementById('heroFundedAmount');

    if (!data.success || !data.reports || data.reports.length === 0) {
      allReportsCache = [];
      container.innerHTML = `
        <div class="empty-board-card" style="padding:48px 24px; text-align:center; background:#fff; border:1px dashed var(--line); border-radius:6px; margin:10px 0;">
          <div style="font-size:36px; margin-bottom:12px;">📋</div>
          <h4 style="font-size:16px; margin-bottom:6px; color:var(--ink);">No civic issues reported yet</h4>
          <p style="font-size:13px; color:var(--slate); max-width:360px; margin:0 auto 16px auto; line-height:1.5;">Be the first citizen to report a road defect, broken streetlight, or sanitation issue in your ward.</p>
          <button type="button" class="fund-btn" onclick="goToReportFlow()" style="margin:0 auto; padding:8px 18px; font-size:12px; width:auto; display:inline-block;">Report a problem now →</button>
        </div>
      `;
      const detail = document.getElementById('detailPanel');
      if (detail) detail.style.display = 'none';

      if (filterCategory === 'all') {
        if (heroCount) heroCount.textContent = '0';
        if (heroFunded) heroFunded.textContent = '₹0';
      }
      renderMapPins([], 'boardMap');
      if (filterCategory === 'all') renderMapPins([], 'heroMap');
      return;
    }

    allReportsCache = data.reports;

    // Calculate total raised across all reports
    let totalRaised = 0;
    data.reports.forEach(r => { totalRaised += parseFloat(r.raised_amount || 0); });

    // Update hero stats
    if (heroCount && filterCategory === 'all') heroCount.textContent = data.reports.length;
    if (heroFunded && filterCategory === 'all') {
      heroFunded.textContent = totalRaised >= 1000
        ? `₹${(totalRaised / 1000).toFixed(1)}K`
        : `₹${totalRaised.toLocaleString('en-IN')}`;
    }

    // Render cards
    container.innerHTML = data.reports.map((r, idx) => {
      const isSelected = (selectedProblemId === r.id || (!selectedProblemId && idx === 0));
      if (isSelected && !selectedProblemId) selectedProblemId = r.id;

      const raised = parseFloat(r.raised_amount || 0);
      const goal = parseFloat(r.goal_amount || 8000);
      const pct = Math.min(100, Math.round((raised / goal) * 100));
      const icon = getCategoryIcon(r.category);
      const bg = getCategoryColor(r.category);

      return `
        <div class="problem-card ${isSelected ? 'selected' : ''}" data-id="${r.id}" onclick="selectProblem('${r.id}')">
          <div class="pcard-thumb" style="background:${bg};">${icon}</div>
          <div class="pcard-body">
            <div class="pcard-top">
              <h4>${r.title}</h4>
              <span class="badge ${r.severity?.toLowerCase() === 'high' ? 'high' : (r.severity?.toLowerCase() === 'low' ? 'verified' : 'mid')}">${r.severity || 'Medium'}</span>
            </div>
            <div class="pcard-loc" style="display:flex; justify-content:space-between; align-items:center;">
              <span>${r.location_text}</span>
              ${getStatusBadgeHtml(r.status)}
            </div>
            <div class="paving-bar">
              <div class="paving-fill" style="width:${pct}%;"></div>
            </div>
            <div class="pcard-foot">
              <span>₹${raised.toLocaleString()} / ₹${goal.toLocaleString()} (${pct}%)</span>
              <span>${r.funder_count || 0} funders</span>
            </div>
            ${r.status === 'FUNDING' || r.status === 'VERIFIED' ? `
            <button class="fund-btn" style="margin-top:8px; width:100%; padding:7px 0; font-size:12px; cursor:pointer;" onclick="event.stopPropagation(); selectProblem('${r.id}'); document.getElementById('detailPanel')?.scrollIntoView({behavior:'smooth', block:'nearest'});">Fund / Contribute</button>` : ''}
          </div>
        </div>
      `;
    }).join('');

    // Update map pins on both board and hero
    renderMapPins(data.reports, 'boardMap');
    if (filterCategory === 'all') {
      renderMapPins(data.reports, 'heroMap');
    }

    // Select first problem or current
    if (selectedProblemId) {
      selectProblem(selectedProblemId);
    }

    if (callback) callback();

  } catch (err) {
    console.error("Board error:", err);
    container.innerHTML = `<div style="padding:30px; text-align:center; color:#E08268;">Failed to load problem board from server.</div>`;
  }
}

function filterBoard(cat, pillElem) {
  document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
  if (pillElem) pillElem.classList.add('active');
  loadProblemBoard(cat);
}

// =========================================================================
// LEAFLET MAP ENGINE (Real India Map)
// =========================================================================

const _leafletMaps = {}; // store initialized map instances

const INDIA_CENTER = [20.5937, 78.9629];
const INDIA_ZOOM   = 5;

function _severityToColor(severity) {
  if (!severity) return '#F2A93B';
  const s = severity.toLowerCase();
  if (s === 'high')   return '#C0503A';
  if (s === 'low')    return '#3C8361';
  return '#F2A93B';
}

function _initLeafletMap(canvasId) {
  const el = document.getElementById(canvasId);
  if (!el || typeof L === 'undefined') return null;

  // Destroy existing instance if any
  if (_leafletMaps[canvasId]) {
    try { _leafletMaps[canvasId].remove(); } catch(e) {}
    delete _leafletMaps[canvasId];
  }

  const map = L.map(canvasId, {
    center: INDIA_CENTER,
    zoom: INDIA_ZOOM,
    zoomControl: true,
    attributionControl: false,
    scrollWheelZoom: canvasId === 'heroMap' ? false : true
  });

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '© OpenStreetMap'
  }).addTo(map);

  _leafletMaps[canvasId] = map;
  return map;
}

function renderMapPins(reports, canvasId = 'boardMap') {
  const el = document.getElementById(canvasId);
  if (!el || typeof L === 'undefined') return;

  // Initialize map if not already done
  let map = _leafletMaps[canvasId];
  if (!map) {
    map = _initLeafletMap(canvasId);
    if (!map) return;
  }

  // Clear old markers
  map.eachLayer(layer => {
    if (layer instanceof L.Marker || layer instanceof L.CircleMarker) {
      map.removeLayer(layer);
    }
  });

  if (!reports || reports.length === 0) {
    map.setView(INDIA_CENTER, INDIA_ZOOM);
    return;
  }

  const validReports = reports.filter(r => r.latitude && r.longitude);

  if (validReports.length === 0) {
    map.setView(INDIA_CENTER, INDIA_ZOOM);
    return;
  }

  const bounds = [];

  validReports.forEach(r => {
    const lat = parseFloat(r.latitude);
    const lng = parseFloat(r.longitude);
    if (isNaN(lat) || isNaN(lng)) return;

    const color = _severityToColor(r.severity);
    const isPulse = r.status === 'IN_PROGRESS' || r.status === 'SUBMITTED_FOR_VERIFICATION';

    const marker = L.circleMarker([lat, lng], {
      radius: 10,
      fillColor: color,
      color: '#fff',
      weight: 2,
      opacity: 1,
      fillOpacity: isPulse ? 1 : 0.85,
      className: isPulse ? 'leaflet-pulse-marker' : ''
    }).addTo(map);

    marker.bindPopup(`
      <div style="font-family:'Inter',sans-serif; font-size:13px; min-width:180px;">
        <b style="display:block; margin-bottom:4px;">${r.title || 'Civic Issue'}</b>
        <span style="font-size:11px; opacity:0.7;">${r.location_text || ''}</span><br>
        <span style="font-size:11px; font-weight:600; color:${color};">${r.severity || 'Medium'} severity</span>
        &nbsp;·&nbsp;
        <span style="font-size:11px;">${(r.status || '').replace(/_/g,' ')}</span>
      </div>
    `, { maxWidth: 240 });

    marker.on('click', () => selectProblem(r.id));
    bounds.push([lat, lng]);
  });

  if (bounds.length === 1) {
    map.setView(bounds[0], 13);
  } else if (bounds.length > 1) {
    map.fitBounds(bounds, { padding: [40, 40], maxZoom: 14 });
  }

  // Force tile refresh after potential visibility change
  setTimeout(() => map.invalidateSize(), 100);
}

async function selectProblem(reportId) {
  selectedProblemId = reportId;

  // Highlight card
  document.querySelectorAll('.problem-card').forEach(c => {
    c.classList.toggle('selected', c.dataset.id === reportId);
  });

  const panel = document.getElementById('detailPanel');
  if (!panel) return;

  try {
    const res = await fetch(`${API_BASE}/api/reports/${reportId}`);
    const data = await res.json();
    if (!data.success || !data.report) return;

    const r = data.report;
    document.getElementById('detailKicker').innerHTML = `${escapeHtml((r.location_text || '').toUpperCase())} &nbsp;·&nbsp; ${getStatusBadgeHtml(r.status)}`;
    document.getElementById('detailTitle').textContent = r.title;

    const badge = document.getElementById('detailBadge');
    badge.textContent = `${r.severity || 'Medium'} severity`;
    badge.className = 'badge ' + (r.severity?.toLowerCase() === 'high' ? 'high' : (r.severity?.toLowerCase() === 'low' ? 'verified' : 'mid'));

    const raised = parseFloat(r.raised_amount || 0);
    const goal = parseFloat(r.goal_amount || 8000);
    const pct = Math.min(100, Math.round((raised / goal) * 100));
    const remaining = Math.max(0, goal - raised);

    document.getElementById('detailFundingRatio').textContent = `₹${raised.toLocaleString()} / ₹${goal.toLocaleString()}`;
    const pctElem = document.getElementById('detailFundingPct');
    pctElem.textContent = `${pct}% funded`;
    if (pct >= 100) {
      pctElem.innerHTML = `<span class="fund-celebrate">✓ Fully Funded</span>`;
    }

    // Animate progress bar fill smoothly
    const progBar = document.getElementById('detailProgressBar');
    progBar.style.width = `${pct}%`;

    document.getElementById('detailRemaining').textContent = `₹${remaining.toLocaleString()} remaining`;
    document.getElementById('detailFundersCount').textContent = `${r.funder_count || 0} funders`;

    document.getElementById('detailCost').textContent = `₹${goal.toLocaleString()}`;
    document.getElementById('detailReleased').textContent = `₹${parseFloat(r.released_amount || 0).toLocaleString()}`;
    document.getElementById('detailEta').textContent = `${r.eta_days || 4} days (${r.status.replace(/_/g, ' ')})`;

    // Transactions list
    const contribList = document.getElementById('contribList');
    if (contribList) {
      if (r.transactions && r.transactions.length > 0) {
        contribList.innerHTML = r.transactions.map(t => `
          <div class="contrib-row">
            <span class="who">${t.donor_name}</span>
            <span style="font-family:'IBM Plex Mono',monospace; font-weight:700; color:var(--green);">+₹${parseFloat(t.amount).toLocaleString()}</span>
          </div>
        `).join('');
      } else {
        contribList.innerHTML = `<div style="padding:10px 0; font-size:12px; opacity:0.6;">Be the first neighbor to back this fix.</div>`;
      }
    }

    panel.classList.add('show');

  } catch (err) {
    console.error("Select problem error:", err);
  }
}

// Preset contribution chips
function selectDemoAmount(amt, chipElem) {
  document.querySelectorAll('.amount-chip').forEach(c => c.classList.remove('active'));
  if (chipElem) chipElem.classList.add('active');
  const input = document.getElementById('customAmountInput');
  if (input) input.value = amt;
}

// Live simulated contribution
async function contributeToSelectedProblem() {
  if (!selectedProblemId) {
    alert("Please select a problem card first.");
    return;
  }

  const amtInput = document.getElementById('customAmountInput');
  const amount = parseFloat(amtInput?.value || 1000);
  if (isNaN(amount) || amount <= 0) {
    alert("Please enter a valid positive contribution amount.");
    return;
  }

  const btn = document.getElementById('btnContributeNow');
  const oldText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Processing demo backing…";

  try {
    const res = await fetch(`${API_BASE}/api/funding/${selectedProblemId}/contribute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        amount: amount,
        donor_name: `${currentUserName} (${currentActivePortal})`,
        user_id: currentUserId,
        note: "Community platform demo contribution"
      })
    });

    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Failed to add contribution");

    btn.textContent = `✓ Backed with ₹${amount.toLocaleString()}!`;

    // Refresh detail panel & card smoothly
    setTimeout(() => {
      btn.disabled = false;
      btn.textContent = oldText;
      selectProblem(selectedProblemId);
      loadProblemBoard();
      loadRewards();
    }, 600);

  } catch (err) {
    alert("Contribution error: " + err.message);
    btn.disabled = false;
    btn.textContent = oldText;
  }
}

// =========================================================================
// 5. CITIZEN MY REPORTS & VISUAL TIMELINE
// =========================================================================

function buildTimelineHtml(currentStatus) {
  const steps = [
    { key: "REPORT", label: "Reported" },
    { key: "FUNDING", label: "In Funding" },
    { key: "FUNDED", label: "Funded" },
    { key: "ASSIGNED", label: "Assigned" },
    { key: "IN_PROGRESS", label: "Work Started" },
    { key: "SUBMITTED_FOR_VERIFICATION", label: "Verification" },
    { key: "COMPLETED", label: "Completed" }
  ];

  const statusOrder = {
    'REPORT': 0,
    'FUNDING': 1,
    'FUNDED': 2,
    'ASSIGNED': 3,
    'IN_PROGRESS': 4,
    'SUBMITTED_FOR_VERIFICATION': 5,
    'VERIFIED': 5,
    'REWORK': 4,
    'COMPLETED': 6
  };

  const currIdx = statusOrder[currentStatus] !== undefined ? statusOrder[currentStatus] : 0;

  return `
    <div class="timeline-stepper">
      ${steps.map((s, idx) => {
        const isDone = idx < currIdx || currentStatus === 'COMPLETED';
        const isActive = idx === currIdx && currentStatus !== 'COMPLETED';
        const icon = isDone ? '✓' : (idx + 1);
        return `
          <div class="t-step ${isDone ? 'done' : ''} ${isActive ? 'active' : ''}">
            <div class="t-dot">${icon}</div>
            <div class="t-label">${s.label}</div>
          </div>
        `;
      }).join('')}
    </div>
  `;
}

async function loadMyReports() {
  const container = document.getElementById('myReportsList');
  if (!container) return;

  try {
    const res = await fetch(`${API_BASE}/api/reports?user_id=usr_citizen_1`);
    const data = await res.json();

    if (!data.success || !data.reports || data.reports.length === 0) {
      container.innerHTML = `
        <div style="background:#fff; padding:30px; border-radius:6px; border:1px solid var(--line); text-align:center;">
          <p style="color:var(--slate); margin-bottom:12px;">You haven't reported any civic defects yet.</p>
          <button class="btn btn-primary" onclick="goToReportFlow()">Report a problem now →</button>
        </div>
      `;
      return;
    }

    container.innerHTML = data.reports.map(r => {
      const raised = parseFloat(r.raised_amount || 0);
      const goal = parseFloat(r.goal_amount || 8000);
      const pct = Math.min(100, Math.round((raised / goal) * 100));

      return `
        <div class="timeline-wrap">
          <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:14px; gap:16px; flex-wrap:wrap;">
            <div>
              <div style="font-size:12px; font-family:'IBM Plex Mono',monospace; opacity:0.65; margin-bottom:4px;">
                REPORT ID: ${r.id} · ${r.location_text.toUpperCase()}
              </div>
              <h3 style="font-size:18px;">${r.title}</h3>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
              ${getStatusBadgeHtml(r.status)}
              <button class="btn btn-ghost" style="padding:6px 14px; font-size:12px;" onclick="selectProblem('${r.id}'); document.getElementById('board').scrollIntoView({behavior:'smooth'});">View Live Details →</button>
            </div>
          </div>

          <!-- Real Database Timeline Stepper -->
          ${buildTimelineHtml(r.status)}

          <!-- Progress summary strip -->
          <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:16px; padding-top:14px; border-top:1px solid var(--line); font-size:12.5px;">
            <div><span style="opacity:0.6; display:block; font-family:'IBM Plex Mono',monospace; font-size:11px;">FUNDING</span><b>₹${raised.toLocaleString()} / ₹${goal.toLocaleString()} (${pct}%)</b></div>
            <div><span style="opacity:0.6; display:block; font-family:'IBM Plex Mono',monospace; font-size:11px;">CONTRACTOR</span><b>${r.contributor_name || 'Awaiting assignment'}</b></div>
            <div><span style="opacity:0.6; display:block; font-family:'IBM Plex Mono',monospace; font-size:11px;">WORK PROGRESS</span><b style="color:var(--green);">${r.latest_progress_pct || 0}% completed</b></div>
            <div><span style="opacity:0.6; display:block; font-family:'IBM Plex Mono',monospace; font-size:11px;">FUNDS DISBURSED</span><b>₹${parseFloat(r.released_amount || 0).toLocaleString()}</b></div>
          </div>
        </div>
      `;
    }).join('');

  } catch (err) {
    console.error("My reports error:", err);
  }
}

// =========================================================================
// 6. CONTRIBUTOR PORTAL WORKFLOW & TASK EXECUTION
// =========================================================================

let currentContributorSubRole = "contractor";

function switchContributorSubRole(subRole, tabElem) {
  currentContributorSubRole = subRole;
  document.querySelectorAll('[data-contrib-role]').forEach(t => t.classList.remove('active'));
  if (tabElem) tabElem.classList.add('active');

  const heading = document.getElementById('contribHeading');
  const sub = document.getElementById('contribSub');
  if (subRole === 'contractor') {
    heading.textContent = "Civil & Road Repair Jobs (Ranked by Suitability)";
    sub.textContent = "Contractors can claim fully funded or in-funding potholes and civil works. Match scores provide transparent fitness recommendation.";
  } else if (subRole === 'engineer') {
    heading.textContent = "Electrical & Drainage Engineering Tasks";
    sub.textContent = "Certified engineers evaluate streetlights, sub-stations, and monsoon stormwater drainage culverts.";
  } else if (subRole === 'ngo') {
    heading.textContent = "Community & Environmental Drives";
    sub.textContent = "Community sanitation drives, waterlogging hotspots, and garbage removal campaigns flagged for institutional backing.";
  } else if (subRole === 'company') {
    heading.textContent = "Corporate Social Responsibility (CSR) Ward Adoptions";
    sub.textContent = "Fund verified municipal bundles with 100% photographic proof and audited public ledger disbursements.";
  }

  loadContributorTasks(subRole);
  loadContributorPayments(subRole);
}

async function loadContributorTasks(role = "contractor") {
  const grid = document.getElementById('contributorTasksGrid');
  if (!grid) return;

  grid.innerHTML = `<div style="padding:40px; text-align:center; color:var(--slate);">Loading field tasks for ${role}...</div>`;

  try {
    const res = await fetch(`${API_BASE}/api/contributor/tasks?role=${encodeURIComponent(role)}`);
    const data = await res.json();

    if (!data.success || !data.tasks || data.tasks.length === 0) {
      grid.innerHTML = `<div style="padding:40px; text-align:center; color:var(--slate); background:#fff; border:1px dashed var(--line); border-radius:6px; grid-column:1/-1;">No field tasks available in this category yet. When citizen reports are verified or reach funding, matching tasks appear here.</div>`;
      return;
    }

    grid.innerHTML = data.tasks.map(t => {
      const isAssigned = (t.status === 'ASSIGNED' || t.status === 'IN_PROGRESS' || t.status === 'SUBMITTED_FOR_VERIFICATION');
      const isMine = (t.contributor_name && t.contributor_name.toLowerCase().includes("ramesh") && role === 'contractor')
                  || (t.contributor_name && t.contributor_name.toLowerCase().includes("vikrant") && role === 'engineer')
                  || (t.contributor_name && t.contributor_name.toLowerCase().includes("sewa") && role === 'ngo');

      let actionBtn = "";
      if (t.status === 'COMPLETED') {
        actionBtn = `<button class="accept-btn" style="background:var(--green); color:#fff;" disabled>✓ Completed & Closed</button>`;
      } else if (t.status === 'SUBMITTED_FOR_VERIFICATION') {
        actionBtn = `<button class="accept-btn" style="background:var(--marigold); color:var(--ink);" disabled>⏳ Awaiting Admin Verification</button>`;
      } else if (isMine && t.status === 'IN_PROGRESS') {
        actionBtn = `<button class="accept-btn" style="background:var(--green); color:#fff;" onclick="openProgressModal('${t.id}', '${t.assignment_id || ''}', '${escapeHtml(t.title)}')">Update Progress / Proof (${t.latest_progress_pct || 50}%) →</button>`;
      } else if (isMine && t.status === 'ASSIGNED') {
        actionBtn = `<button class="accept-btn" style="background:var(--ink); color:#fff;" onclick="startWorkOnTask('${t.id}')">Start Field Work Now →</button>`;
      } else if (!isAssigned) {
        actionBtn = `<button class="accept-btn" onclick="acceptTask('${t.id}')">Accept Job (${t.match_score}% fit) →</button>`;
      } else {
        actionBtn = `<button class="accept-btn" style="opacity:0.6;" onclick="acceptTask('${t.id}')">Re-assign Job to Me</button>`;
      }

      return `
        <div class="job-card">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span class="tag" style="background:rgba(60,131,97,0.18); color:var(--green); font-weight:700;">${t.match_score}% match</span>
            ${getStatusBadgeHtml(t.status)}
          </div>
          <h4>${t.title}</h4>
          <p style="margin-bottom:8px;">${t.work_scope || 'Defect rehabilitation scoped by AI vision system.'}</p>
          <div class="job-meta">
            <span>${t.distance_km} KM AWAY</span>
            <span>₹${parseFloat(t.budget).toLocaleString()} BUDGET</span>
          </div>
          ${t.contributor_name ? `<div style="font-size:11.5px; margin-bottom:10px; opacity:0.8; font-family:'IBM Plex Mono',monospace;">ASSIGNED: <b>${t.contributor_name}</b></div>` : ''}
          ${actionBtn}
        </div>
      `;
    }).join('');

  } catch (err) {
    console.error("Contributor tasks error:", err);
  }
}

function escapeHtml(text) {
  if (!text) return "";
  return text.replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

async function acceptTask(reportId) {
  const userSelect = document.getElementById('userSelector');
  const contribId = userSelect?.value || 'usr_contractor_1';
  const contribName = userSelect?.options[userSelect.selectedIndex]?.text?.split('(')[0]?.trim() || "Ramesh Contractors";

  try {
    const res = await fetch(`${API_BASE}/api/tasks/${reportId}/accept`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contributor_id: contribId,
        contributor_name: contribName,
        contributor_role: currentContributorSubRole
      })
    });

    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Failed to accept task");

    alert(`Job accepted! Status updated to ASSIGNED.`);
    loadContributorTasks(currentContributorSubRole);
  } catch (err) {
    alert("Error accepting task: " + err.message);
  }
}

async function startWorkOnTask(reportId) {
  try {
    const res = await fetch(`${API_BASE}/api/tasks/${reportId}/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contributor_id: currentUserId })
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Failed to start task");

    alert("Ground work marked as STARTED on site!");
    loadContributorTasks(currentContributorSubRole);
  } catch (err) {
    alert("Error starting work: " + err.message);
  }
}

// Progress modal handling
let currentProgressPhotoFile = null;

function openProgressModal(reportId, assignmentId, title) {
  document.getElementById('progModalReportId').value = reportId;
  document.getElementById('progModalAssignmentId').value = assignmentId || "";
  document.getElementById('progModalTitle').textContent = `Update Progress: ${title}`;
  document.getElementById('progWorkNote').value = "";
  document.getElementById('progPhotoPreview').style.display = 'none';
  document.getElementById('progPhotoPlaceholder').style.display = 'block';
  currentProgressPhotoFile = null;

  document.getElementById('modalProgressUpdate').classList.add('show');
}

function closeProgressModal() {
  document.getElementById('modalProgressUpdate').classList.remove('show');
}

function previewProgressPhoto(input) {
  if (input.files && input.files[0]) {
    currentProgressPhotoFile = input.files[0];
    const preview = document.getElementById('progPhotoPreview');
    preview.src = URL.createObjectURL(input.files[0]);
    preview.style.display = 'block';
    document.getElementById('progPhotoPlaceholder').style.display = 'none';
  }
}

async function submitProgressUpdate() {
  const reportId = document.getElementById('progModalReportId').value;
  const asgnId = document.getElementById('progModalAssignmentId').value;
  const pcts = document.getElementsByName('progPct');
  let pct = 50;
  for (let r of pcts) {
    if (r.checked) { pct = parseInt(r.value); break; }
  }

  const note = document.getElementById('progWorkNote').value.trim() || `Milestone ${pct}% completed on site.`;
  const submitVerif = document.getElementById('chkSubmitVerification').checked;

  const formData = new FormData();
  formData.append("assignment_id", asgnId);
  formData.append("progress_pct", pct);
  formData.append("work_note", note);
  formData.append("submit_verification", submitVerif ? "true" : "false");

  if (currentProgressPhotoFile) {
    formData.append("image", currentProgressPhotoFile);
  } else {
    // Default fallback progress photo
    formData.append("progress_image", "/uploads/progress/progress_potholes.jpg");
  }

  try {
    const res = await fetch(`${API_BASE}/api/tasks/${reportId}/progress`, {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Failed to submit progress");

    closeProgressModal();
    alert(`Progress milestone (${pct}%) submitted for Municipal Verification!`);
    loadContributorTasks(currentContributorSubRole);
  } catch (err) {
    alert("Error submitting progress: " + err.message);
  }
}

// =========================================================================
// 6b. CONTRIBUTOR DISBURSEMENTS & PAYMENTS LEDGER
// =========================================================================

async function loadContributorPayments(role = 'contractor', contributorId = null) {
  const container = document.getElementById('contributorPaymentsContainer');
  const earningsElem = document.getElementById('contributorTotalEarnings');
  if (!container) return;

  try {
    let url = `${API_BASE}/api/contributor/payments?role=${encodeURIComponent(role)}`;
    if (contributorId) {
      url += `&contributor_id=${encodeURIComponent(contributorId)}`;
    }
    const res = await fetch(url);
    const data = await res.json();

    if (!data.success) throw new Error(data.error || "Failed to load payments");

    if (earningsElem) {
      earningsElem.textContent = `₹${parseFloat(data.total_earned || 0).toLocaleString('en-IN')}`;
    }

    if (!data.payments || data.payments.length === 0) {
      container.innerHTML = `
        <div style="padding:36px; text-align:center; color:var(--slate);">
          <div style="font-size:26px; margin-bottom:8px;">💳</div>
          <b>No disbursements credited yet for ${(role || 'contractor').toUpperCase()}.</b>
          <p style="font-size:12px; margin-top:6px; opacity:0.7;">Milestone payments are automatically credited when administrative verification passes.</p>
        </div>
      `;
      return;
    }

    container.innerHTML = `
      <table class="admin-table" style="margin:0; width:100%; border-collapse:collapse;">
        <thead>
          <tr style="background:var(--paper-dim); border-bottom:1px solid var(--line); font-size:11px; font-family:'IBM Plex Mono',monospace; text-transform:uppercase;">
            <th style="padding:12px 16px; text-align:left;">DISBURSEMENT REF & DATE</th>
            <th style="padding:12px 16px; text-align:left;">CIVIC DEFECT & LOCATION</th>
            <th style="padding:12px 16px; text-align:left;">CREDITED ENTITY</th>
            <th style="padding:12px 16px; text-align:left;">VERIFICATION NOTES</th>
            <th style="padding:12px 16px; text-align:right;">AMOUNT RELEASED</th>
          </tr>
        </thead>
        <tbody>
          ${data.payments.map(p => `
            <tr style="border-bottom:1px solid var(--line); font-size:12.5px;">
              <td style="padding:12px 16px;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:700; color:var(--ink-soft);">${escapeHtml(p.id)}</div>
                <div style="font-size:11px; opacity:0.6;">${escapeHtml(p.created_at || 'Recently')}</div>
              </td>
              <td style="padding:12px 16px;">
                <b>${escapeHtml(p.report_title || 'Civic Repair')}</b>
                <div style="font-size:11px; opacity:0.7;">${escapeHtml(p.location_text || '')}</div>
              </td>
              <td style="padding:12px 16px;">
                <span style="font-weight:600;">${escapeHtml(p.contributor_name || 'Contributor')}</span>
                <div style="font-size:11px; font-family:'IBM Plex Mono',monospace; opacity:0.6; text-transform:uppercase;">${escapeHtml(p.contributor_role || role)}</div>
              </td>
              <td style="padding:12px 16px; font-size:11.5px; color:var(--slate); max-width:240px;">
                ${escapeHtml(p.notes || 'Verified compliant with municipal engineering standards.')}
              </td>
              <td style="padding:12px 16px; text-align:right; font-family:'IBM Plex Mono',monospace; font-weight:800; font-size:14px; color:var(--green);">
                +₹${parseFloat(p.amount_released || 0).toLocaleString('en-IN')}
              </td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  } catch (err) {
    console.error("Contributor payments error:", err);
    container.innerHTML = `<div style="padding:24px; text-align:center; color:var(--terracotta);">Failed to load financial records.</div>`;
  }
}

// =========================================================================
// 7. ADMIN VERIFICATION & CENTRAL CONTROL
// =========================================================================

function switchAdminTab(tab, tabElem) {
  document.querySelectorAll('.admin-tab').forEach(t => t.classList.remove('active'));
  if (tabElem) tabElem.classList.add('active');

  document.getElementById('adminTabQueue').style.display = (tab === 'queue') ? 'block' : 'none';
  document.getElementById('adminTabReports').style.display = (tab === 'reports') ? 'block' : 'none';
  document.getElementById('adminTabHistory').style.display = (tab === 'history') ? 'block' : 'none';

  if (tab === 'queue') loadAdminVerifications();
  if (tab === 'reports') loadAdminReportsTable();
  if (tab === 'history') loadAdminHistoryFeed();
}

async function loadAdminDashboard() {
  try {
    const res = await fetch(`${API_BASE}/api/admin/dashboard`);
    const data = await res.json();
    if (!data.success || !data.stats) return;

    const s = data.stats;
    document.getElementById('kpiReports').textContent = s.total_reports || 0;
    document.getElementById('kpiFunding').textContent = s.funding_reports || 0;
    document.getElementById('kpiInProgress').textContent = (s.in_progress_jobs || 0) + (s.assigned_jobs || 0);
    document.getElementById('kpiPendingVerif').textContent = s.pending_verification || 0;
    document.getElementById('kpiFundsReleased').textContent = `₹${parseFloat(s.total_funds_released || 0).toLocaleString()}`;
    document.getElementById('tabPendingCount').textContent = s.pending_verification || 0;
  } catch (err) {
    console.error("Dashboard KPI error:", err);
  }
}

async function loadAdminVerifications() {
  const container = document.getElementById('adminVerifCardsContainer');
  if (!container) return;

  try {
    const res = await fetch(`${API_BASE}/api/admin/verifications?status=pending`);
    const data = await res.json();

    if (!data.success || !data.verifications || data.verifications.length === 0) {
      container.innerHTML = `<div style="background:#fff; padding:30px; text-align:center; border:1px solid var(--line); border-radius:6px; color:var(--slate);">✓ No pending work verifications in queue. All field updates are verified.</div>`;
      return;
    }

    container.innerHTML = data.verifications.map(v => {
      const origImg = v.original_image || "/uploads/reports/sample_potholes.jpg";
      const proofImg = v.progress_image || "/uploads/progress/progress_potholes.jpg";
      const suggestedRelease = parseFloat(v.suggested_release_amount || 0);

      return `
        <div class="verify-card">
          <div>
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
              <div>
                <span class="st-badge st-SUBMITTED_FOR_VERIFICATION">Awaiting Approval</span>
                <h3 style="font-size:18px; margin-top:6px;">${v.title}</h3>
                <div style="font-size:12px; color:var(--slate); margin-top:2px;">
                  ${v.location_text} · Submitted by <b>${v.contributor_name || 'Contractor'}</b> (${v.contributor_role || 'field crew'})
                </div>
              </div>
              <span class="badge ${v.severity?.toLowerCase() === 'high' ? 'high' : 'mid'}">${v.severity || 'Medium'}</span>
            </div>

            <!-- Before vs After Photos -->
            <div class="verify-photos">
              <div class="verify-photo-box">
                <img src="${origImg}" alt="Original defect" />
                <span>BEFORE: ORIGINAL REPORT</span>
              </div>
              <div class="verify-photo-box">
                <img src="${proofImg}" alt="Contractor work proof" />
                <span>AFTER: ${v.progress_pct}% WORK PROOF</span>
              </div>
            </div>

            <div style="margin-top:12px; padding:10px 14px; background:var(--paper-dim); border-radius:4px; font-size:12.5px;">
              <b>Contractor Note:</b> <i>"${v.work_note || 'Milestone proof submitted for inspection.'}"</i>
            </div>
          </div>

          <!-- Financials & Decision Actions -->
          <div style="border-left:1px solid var(--line); padding-left:24px; display:flex; flex-direction:column; justify-content:space-between;">
            <div>
              <div class="result-label" style="margin-bottom:6px;">FINANCIAL LEDGER</div>
              <div style="font-size:13px; margin-bottom:4px;">CAMPAIGN GOAL: <b>₹${parseFloat(v.goal_amount || 0).toLocaleString()}</b></div>
              <div style="font-size:13px; margin-bottom:4px;">TOTAL RAISED: <b style="color:var(--marigold-deep);">₹${parseFloat(v.raised_amount || 0).toLocaleString()}</b></div>
              <div style="font-size:13px; margin-bottom:14px;">ALREADY RELEASED: <b style="color:var(--green);">₹${parseFloat(v.released_amount || 0).toLocaleString()}</b></div>

              <div style="background:rgba(60,131,97,0.1); border:1px solid rgba(60,131,97,0.25); padding:10px 14px; border-radius:4px; margin-bottom:18px;">
                <span style="font-size:11px; font-family:'IBM Plex Mono',monospace; color:var(--green); font-weight:700;">SUGGESTED MILESTONE DISBURSEMENT:</span>
                <div style="font-size:20px; font-family:'Archivo'; font-weight:800; color:var(--green); margin-top:2px;">₹${suggestedRelease.toLocaleString()}</div>
              </div>
            </div>

            <div style="display:flex; flex-direction:column; gap:8px;">
              <button class="btn btn-primary" onclick="openAdminVerifyModal('${v.update_id}', '${v.report_id}', '${escapeHtml(v.title)}', '${origImg}', '${proofImg}', ${v.goal_amount || 0}, ${v.raised_amount || 0}, ${v.released_amount || 0}, ${suggestedRelease})">
                ✓ Review & Disburse Funds
              </button>
            </div>
          </div>
        </div>
      `;
    }).join('');

  } catch (err) {
    console.error("Verifications queue error:", err);
  }
}

function openAdminVerifyModal(updateId, reportId, title, origImg, proofImg, goal, raised, released, suggested) {
  document.getElementById('admModalUpdateId').value = updateId;
  document.getElementById('admModalReportId').value = reportId;
  document.getElementById('admModalTitle').textContent = `Verify: ${title}`;
  document.getElementById('admModalOriginalImg').src = origImg;
  document.getElementById('admModalProofImg').src = proofImg;

  document.getElementById('admModalGoal').textContent = `₹${parseFloat(goal).toLocaleString()}`;
  document.getElementById('admModalRaised').textContent = `₹${parseFloat(raised).toLocaleString()}`;
  document.getElementById('admModalReleased').textContent = `₹${parseFloat(released).toLocaleString()}`;
  document.getElementById('admReleaseAmount').value = suggested || 0;
  document.getElementById('admNotes').value = "Work verified compliant with Municipal Corporation technical standards.";

  document.getElementById('modalAdminVerify').classList.add('show');
}

function closeAdminVerifyModal() {
  document.getElementById('modalAdminVerify').classList.remove('show');
}

async function adminTriggerApprove() {
  const updateId = document.getElementById('admModalUpdateId').value;
  const reportId = document.getElementById('admModalReportId').value;
  const amount = parseFloat(document.getElementById('admReleaseAmount').value || 0);
  const notes = document.getElementById('admNotes').value.trim();

  try {
    const res = await fetch(`${API_BASE}/api/admin/verifications/${updateId}/approve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        report_id: reportId,
        admin_id: "usr_admin_1",
        notes: notes,
        amount_to_release: amount
      })
    });

    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Failed to approve verification");

    closeAdminVerifyModal();
    alert(`✓ Verification approved! Released ₹${amount.toLocaleString()} to contractor. Report status updated to: ${data.status}`);

    loadAdminDashboard();
    loadAdminVerifications();
    loadAdminReportsTable();
  } catch (err) {
    alert("Approval error: " + err.message);
  }
}

async function adminTriggerRework() {
  const updateId = document.getElementById('admModalUpdateId').value;
  const reportId = document.getElementById('admModalReportId').value;
  const notes = document.getElementById('admNotes').value.trim() || "On-site quality check failed. Rework requested.";

  try {
    const res = await fetch(`${API_BASE}/api/admin/verifications/${updateId}/rework`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        report_id: reportId,
        admin_id: "usr_admin_1",
        notes: notes
      })
    });

    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Failed to flag rework");

    closeAdminVerifyModal();
    alert("Case flagged for REWORK. Contractor notified.");

    loadAdminDashboard();
    loadAdminVerifications();
    loadAdminReportsTable();
  } catch (err) {
    alert("Rework error: " + err.message);
  }
}

async function loadAdminReportsTable() {
  const tbody = document.getElementById('adminReportsTableBody');
  if (!tbody) return;

  try {
    const res = await fetch(`${API_BASE}/api/admin/reports`);
    const data = await res.json();
    if (!data.success || !data.reports || data.reports.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:30px; color:var(--slate);">No reports in the system. Reports logged via the citizen portal will appear here.</td></tr>`;
      return;
    }

    tbody.innerHTML = data.reports.map(r => {
      const raised = parseFloat(r.raised_amount || 0);
      const goal = parseFloat(r.goal_amount || 8000);
      const released = parseFloat(r.released_amount || 0);

      return `
        <tr>
          <td>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:11px; opacity:0.6;">${r.id}</div>
            <b>${r.title}</b>
            <div style="font-size:11.5px; opacity:0.7;">${r.location_text}</div>
          </td>
          <td><span style="font-family:'IBM Plex Mono',monospace;">${r.category}</span></td>
          <td><span class="badge ${r.severity?.toLowerCase() === 'high' ? 'high' : 'mid'}">${r.severity || 'Medium'}</span></td>
          <td>${getStatusBadgeHtml(r.status)}</td>
          <td><b>₹${raised.toLocaleString()}</b> / ₹${goal.toLocaleString()}</td>
          <td style="color:var(--green); font-weight:700;">₹${released.toLocaleString()}</td>
          <td>${r.contributor_name || '<i style="opacity:0.5;">Unassigned</i>'}</td>
          <td>
            <div style="display:flex; gap:6px;">
              <button class="btn btn-ghost" style="padding:4px 10px; font-size:11px;" onclick="selectProblem('${r.id}'); switchPortal('citizen');">Inspect</button>
              <button class="btn" style="padding:4px 8px; font-size:11px; background:#fff0ed; color:var(--terracotta); border:1px solid #f5c2b8; border-radius:4px; cursor:pointer;" onclick="adminDeleteReport('${r.id}', '${escapeHtml(r.title)}')">Delete</button>
            </div>
          </td>
        </tr>
      `;
    }).join('');

  } catch (err) {
    console.error("Reports table error:", err);
  }
}

async function adminDeleteReport(reportId, reportTitle) {
  const confirmed = confirm(`Are you sure you want to permanently delete this report?\n\n"${reportTitle || reportId}"\n\nThis will remove all associated funding records, assignments, and field verification logs.`);
  if (!confirmed) return;

  try {
    const res = await fetch(`${API_BASE}/api/reports/${reportId}`, {
      method: "DELETE"
    });
    const data = await res.json();
    if (!data.success) throw new Error(data.error || "Failed to delete report");

    alert("✓ Report and associated records permanently deleted.");
    loadAdminDashboard();
    loadAdminReportsTable();
    loadAdminVerifications();
    loadProblemBoard('all');
    loadMyReports();
  } catch (err) {
    alert("Delete failed: " + err.message);
  }
}

async function loadAdminHistoryFeed() {
  const feed = document.getElementById('adminHistoryFeed');
  if (!feed) return;

  feed.innerHTML = '<div style="padding:20px; text-align:center; color:var(--slate); font-size:13px;">Loading audit history...</div>';

  try {
    const res = await fetch(`${API_BASE}/api/admin/history?limit=30`);
    const data = await res.json();

    if (!data.success || !data.history || data.history.length === 0) {
      feed.innerHTML = `
        <div style="padding:30px; text-align:center; color:var(--slate);">
          <b>No audit events yet.</b>
          <p style="font-size:13px; margin-top:6px; opacity:0.7;">Status transitions appear here as reports progress through the civic lifecycle.</p>
        </div>
      `;
      return;
    }

    const statusColors = {
      'REPORT':                     'var(--slate)',
      'FUNDING':                    'var(--marigold-deep)',
      'FUNDED':                     'var(--green)',
      'ASSIGNED':                   'var(--ink-soft)',
      'IN_PROGRESS':                'var(--marigold-deep)',
      'SUBMITTED_FOR_VERIFICATION': 'var(--terracotta)',
      'VERIFIED':                   'var(--green)',
      'REWORK':                     'var(--terracotta)',
      'COMPLETED':                  'var(--green)'
    };

    feed.innerHTML = '<div style="display:flex; flex-direction:column; gap:0;">'
      + data.history.map(h => {
          const color = statusColors[h.to_status] || 'var(--slate)';
          const timeStr = h.created_at
            ? new Date(h.created_at).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })
            : '';
          const actor = h.actor_name ? ` by <b>${h.actor_name}</b> (${h.actor_role || 'system'})` : '';
          const note = h.note ? ` &mdash; <i>${h.note}</i>` : '';
          return `
            <div class="audit-item" style="font-size:13px; line-height:1.6; padding:10px 0; border-bottom:1px dashed var(--line);">
              <span style="color:${color}; font-weight:700;">${(h.to_status || '').replace(/_/g, ' ')}:</span>
              <b>${h.report_title || h.report_id}</b>${actor}${note}
              <span style="float:right; font-family:'IBM Plex Mono',monospace; font-size:11px; opacity:0.5;">${timeStr}</span>
            </div>
          `;
        }).join('')
      + '</div>';

  } catch (err) {
    console.error('History feed error:', err);
    feed.innerHTML = '<div style="padding:20px; color:#E08268; font-size:13px;">Failed to load audit history from server.</div>';
  }
}

// =========================================================================
// 8. REWARDS SYSTEM (DATABASE-DRIVEN)
// =========================================================================

async function loadRewards() {
  try {
    const res = await fetch(`${API_BASE}/api/rewards?user_id=${encodeURIComponent(currentUserId)}`);
    const data = await res.json();
    if (!data.success) return;

    const r = data.rewards || {};
    const scoreElem = document.getElementById('rewScore');
    if (scoreElem) scoreElem.textContent = (r.points || 0).toLocaleString();

    const contribElem = document.getElementById('rewContributed');
    if (contribElem) contribElem.textContent = `₹${parseFloat(r.contributed_amount || 0).toLocaleString()}`;

    const backedElem = document.getElementById('rewBacked');
    if (backedElem) backedElem.textContent = r.problems_backed || 0;

    const tierElem = document.getElementById('rewTier');
    if (tierElem) tierElem.textContent = r.tier || "Bronze";

    // Leaderboard
    const lbContainer = document.getElementById('rewardsLeaderboard');
    if (lbContainer) {
      if (data.leaderboard && data.leaderboard.length > 0) {
        lbContainer.innerHTML = data.leaderboard.map((item, idx) => {
          const rank = String(item.rank || (idx + 1)).padStart(2, '0');
          const initials = item.user_name ? item.user_name.split(' ').map(n => n[0]).join('').slice(0, 2).toUpperCase() : 'CU';
          const roleLabel = item.role ? `<span style="text-transform:capitalize; opacity:0.75; font-size:11px; margin-left:4px;">· ${escapeHtml(item.role)}</span>` : '';
          const fixesCount = item.completed_fixes !== undefined && item.completed_fixes > 0
            ? `${item.completed_fixes} fixes`
            : `${item.problems_backed || 0} backed`;
          return `
            <div class="lb-row">
              <span class="lb-rank">${rank}</span>
              <span class="lb-avatar">${initials}</span>
              <span class="lb-name">
                ${escapeHtml(item.user_name || 'Contributor')}
                ${roleLabel}
                <small style="opacity:0.6; font-size:11px; font-family:'IBM Plex Mono',monospace;">(${escapeHtml(item.tier || 'Silver')})</small>
              </span>
              <span class="lb-score">${fixesCount} · ${(item.points || 0).toLocaleString()} pts</span>
            </div>
          `;
        }).join('');
      } else {
        lbContainer.innerHTML = `
          <div style="padding:24px; text-align:center; color:var(--slate); font-size:13px;">
            No ranked community contributors yet. Back a civic problem or submit reports to earn points and climb the board!
          </div>
        `;
      }
    }

  } catch (err) {
    console.error("Rewards error:", err);
  }
}

// =========================================================================
// 9. EVENT LISTENERS & INITIALIZATION
// =========================================================================

// File uploads
if (fileInput) {
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      showPreview(file, file.name);
      sendImageToAI(file, file.name);
    }
  });
}

if (uploadZone) {
  uploadZone.addEventListener('click', (e) => {
    if (e.target === changePhotoBtn || !uploadPreviewContainer.contains(e.target)) {
      fileInput.click();
    }
  });

  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
  });

  uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
  });

  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      showPreview(file, file.name);
      sendImageToAI(file, file.name);
    }
  });
}

if (changePhotoBtn) {
  changePhotoBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });
}

const locInputField = document.getElementById('locationInput');
if (locInputField) {
  locInputField.addEventListener('change', () => {
    if (currentAIData) fetchAndRenderMatches(currentAIData);
  });
  locInputField.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (currentAIData) fetchAndRenderMatches(currentAIData);
    }
  });
}

// Window load init
window.addEventListener('load', () => {
  checkBackendHealth();
  // Pre-initialize both maps immediately so they're ready
  _initLeafletMap('heroMap');
  _initLeafletMap('boardMap');
  loadProblemBoard('all');
  loadMyReports();
  loadRewards();
});
