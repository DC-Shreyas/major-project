const state = { options: null, overview: null, recordsPage: 1, recordTotalPages: 1, debounce: null };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function money(value, compact = false) {
  const amount = Number(value) || 0;
  if (!amount) return "₹0";
  if (compact && amount >= 10000000) return "₹" + (amount / 10000000).toFixed(2).replace(/0+$/, "").replace(/\.$/, "") + " Cr";
  if (compact && amount >= 100000) return "₹" + (amount / 100000).toFixed(1).replace(/\.0$/, "") + " L";
  if (amount >= 10000000) return "₹" + (amount / 10000000).toFixed(2) + " Cr";
  if (amount >= 100000) return "₹" + (amount / 100000).toFixed(1) + " L";
  return "₹" + Math.round(amount).toLocaleString("en-IN");
}
function number(value) { return (Number(value) || 0).toLocaleString("en-IN"); }
function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]
  );
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "The request could not be completed.");
  return result;
}
function setSelectOptions(select, values, firstLabel = null) {
  const first = firstLabel ? '<option value="">' + escapeHTML(firstLabel) + "</option>" : "";
  select.innerHTML = first + values.map((value) => {
    const label = typeof value === "string" ? value : value.label;
    const key = typeof value === "string" ? value : value.value;
    return '<option value="' + escapeHTML(key) + '">' + escapeHTML(label) + "</option>";
  }).join("");
}
function fillGlobalOptions() {
  const values = state.options;
  setSelectOptions($("#filter-sro"), values.sro_areas, "All areas");
  setSelectOptions($("#filter-type"), values.property_types, "All property types");
  setSelectOptions($("#filter-quarter"), values.quarters, "All quarters");
  setSelectOptions($("#records-sro"), values.sro_areas, "All areas");
  setSelectOptions($("#records-type"), values.property_types, "All types");
  setSelectOptions($("#records-quarter"), values.quarters, "All quarters");
  setSelectOptions($("#input-sro"), values.sro_areas);
  setSelectOptions($("#input-type"), values.property_types);
  setSelectOptions($("#input-quarter"), values.estimate_quarters);
  $("#input-quarter").value = values.default_quarter;
  $("#input-sro").value = values.sro_areas.includes("JP Nagar") ? "JP Nagar" : values.sro_areas[0];
  $("#input-type").value = values.property_types.includes("Apartment/Flat") ? "Apartment/Flat" : values.property_types[0];
  syncMarketOptions(true);
}
function selectedMarket() {
  if (!state.options) return null;
  return state.options.markets.find((item) =>
    item.micro_market_id === $("#input-market").value
    && item.sro_area === $("#input-sro").value
    && item.property_type === $("#input-type").value
  ) || null;
}
function syncMarketOptions(chooseFirst = false) {
  const old = $("#input-market").value;
  const matches = state.options.markets
    .filter((item) => item.sro_area === $("#input-sro").value && item.property_type === $("#input-type").value)
    .sort((a, b) => b.records - a.records || a.micro_market.localeCompare(b.micro_market));
  const unique = [], seen = new Set();
  for (const item of matches) {
    if (!seen.has(item.micro_market_id)) { unique.push(item); seen.add(item.micro_market_id); }
  }
  const options = unique.map((item) => ({
    value: item.micro_market_id,
    label: item.micro_market + " · " + item.records + " records",
  }));
  setSelectOptions($("#input-market"), options, "New / unlisted micro-market");
  if (unique.some((item) => item.micro_market_id === old)) $("#input-market").value = old;
  else if (chooseFirst && unique.length) $("#input-market").value = unique[0].micro_market_id;
  if ($("#input-market").value) fillFromMarket();
  else fillSignalDefaults();
}
function fillFromMarket() {
  const market = selectedMarket();
  if (!market) return fillSignalDefaults();
  $("#input-rate").value = market.guideline_rate_inr_per_sqm;
  $("#input-area").value = market.typical_area_sqft || 1200;
  $("#input-bedrooms").value = market.typical_bedrooms ?? "";
  $("#input-age").value = market.typical_age_years ?? 10;
  $("#input-floor").value = market.typical_bedrooms == null ? "" : 5;
  const guide = market.guideline_rate_inr_per_sqm * Number($("#input-area").value) / 10.7639;
  $("#input-declared").value = Math.max(market.median_declared_price_inr, guide).toFixed(0);
  $("#input-asking").value = Math.max(market.median_asking_price_inr, Number($("#input-declared").value) * 1.05).toFixed(0);
}
function fillSignalDefaults() {
  const rate = Number($("#input-rate").value) || 45000;
  const area = Number($("#input-area").value) || 1200;
  const guide = rate * area / 10.7639;
  $("#input-declared").value = Math.round(guide * 1.12);
  $("#input-asking").value = Math.round(guide * 1.42);
}
function updateSignalDefaults() {
  const market = selectedMarket();
  if (!market) return fillSignalDefaults();
  const area = Number($("#input-area").value);
  const rate = Number($("#input-rate").value);
  if (!(area > 0) || !(rate > 0)) return;
  const scale = area / Math.max(Number(market.typical_area_sqft) || area, 1);
  const guidance = rate * area / 10.7639;
  const declared = Math.max(Number(market.median_declared_price_inr) * scale, guidance);
  const asking = Math.max(Number(market.median_asking_price_inr) * scale, declared * 1.05);
  $("#input-declared").value = Math.round(declared);
  $("#input-asking").value = Math.round(asking);
}
function queryString(values) {
  const params = new URLSearchParams();
  Object.entries(values).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") params.set(key, value);
  });
  return params.toString();
}
function currentOverviewFilters() {
  return { sro: $("#filter-sro").value, type: $("#filter-type").value, quarter: $("#filter-quarter").value };
}
function displayQuarter(value) { return String(value).replace(/^(\d{4})Q(\d)$/, "$1 Q$2"); }

async function updateSummary() {
  try {
    const summary = await api("/api/summary?" + queryString(currentOverviewFilters()));
    state.overview = summary;
    $("#kpi-records").textContent = number(summary.rows);
    $("#kpi-price").textContent = money(summary.median_asking_price_inr, true);
    $("#kpi-sqft").textContent = money(summary.median_asking_price_per_sqft_inr) + " / sq ft";
    $("#kpi-mape").textContent = summary.backtest.mape_percent.toFixed(1) + "%";
    $("#kpi-mape-foot").textContent = "2024 Q1 synthetic holdout · " + number(state.options.report.evaluation_rows) + " rows";
    $("#overview-row-count").textContent = number(summary.rows) + " generated records";
    renderTrend(summary.trend);
    renderMix(summary.property_mix, summary.rows);
    renderMarkets(summary.markets);
  } catch (error) { $("#overview-row-count").textContent = "Unable to load summary"; }
}
function renderTrend(rows) {
  const box = $("#trend-chart");
  if (!rows.length) { box.innerHTML = '<div class="chart-empty">No records match these filters.</div>'; return; }
  const width = Math.max(box.clientWidth, 420), height = 202;
  const left = 48, right = width - 12, top = 13, bottom = height - 27;
  const series = [
    { key: "guidance_price_inr", cls: "reference", color: "#398367" },
    { key: "asking_price_inr", cls: "asking", color: "#d88354" },
    { key: "declared_price_inr", cls: "declared", color: "#8b9aa0" },
  ];
  const all = rows.flatMap((row) => series.map((item) => Number(row[item.key]))).filter(Boolean);
  const min = Math.min(...all) * .88, max = Math.max(...all) * 1.06;
  const x = (index) => rows.length === 1 ? (left + right) / 2 : left + index * (right - left) / (rows.length - 1);
  const y = (value) => bottom - ((value - min) / Math.max(max - min, 1)) * (bottom - top);
  let grid = "";
  for (let index = 0; index < 4; index++) {
    const yy = top + index * (bottom - top) / 3, value = max - index * (max - min) / 3;
    grid += '<line class="trend-grid-line" x1="' + left + '" y1="' + yy + '" x2="' + right + '" y2="' + yy + '"></line>';
    grid += '<text class="trend-label" x="0" y="' + (yy + 3) + '">' + money(value, true) + "</text>";
  }
  let lines = "";
  for (const item of series) {
    const points = rows.map((row, index) => x(index) + "," + y(Number(row[item.key]))).join(" ");
    lines += '<polyline class="trend-line ' + item.cls + '" points="' + points + '"></polyline>';
    rows.forEach((row, index) => {
      lines += '<circle class="trend-point" cx="' + x(index) + '" cy="' + y(Number(row[item.key])) + '" r="4" fill="' + item.color + '"><title>' + escapeHTML(displayQuarter(row.quarter)) + ": " + money(row[item.key]) + "</title></circle>";
    });
  }
  const labels = rows.map((row, index) =>
    '<text class="trend-label" text-anchor="middle" x="' + x(index) + '" y="' + (height - 5) + '">' + escapeHTML(displayQuarter(row.quarter)) + "</text>"
  ).join("");
  box.innerHTML = '<svg class="trend-svg" viewBox="0 0 ' + width + " " + height + '" role="img" aria-label="Median generated price signals by quarter">' + grid + lines + labels + "</svg>";
}
function renderMix(rows, total) {
  const box = $("#property-mix");
  if (!rows.length) { box.innerHTML = '<div class="chart-empty">No property types in this view.</div>'; return; }
  box.innerHTML = rows.map((row) => {
    const percent = total ? row.records / total * 100 : 0;
    return '<div class="mix-item"><div class="mix-item-head"><span>' + escapeHTML(row.property_type) + "</span><strong>" + number(row.records) + '</strong></div><div class="mix-track"><div class="mix-fill" style="width:' + percent.toFixed(1) + '%"></div></div><div class="mix-caption">' + percent.toFixed(1) + "% of filtered records</div></div>";
  }).join("");
}
function renderMarkets(rows) {
  const body = $("#market-table");
  if (!rows.length) { body.innerHTML = '<tr><td class="empty-cell" colspan="4">No micro-markets match these filters.</td></tr>'; return; }
  body.innerHTML = rows.map((row) =>
    "<tr><td>" + escapeHTML(row.micro_market) + '<span class="table-sub">' + escapeHTML(row.sro_area) + "</span></td><td>" + number(row.records) + "</td><td>" + money(row.median_asking_price_per_sqft_inr) + ' <span class="table-sub">per sq ft</span></td><td>' + money(row.median_asking_price_inr, true) + "</td></tr>"
  ).join("");
}
function switchView(name) {
  const titles = { overview: "Overview", records: "Explore records", estimate: "Estimate a home", about: "About this dataset" };
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === "view-" + name));
  $$(".nav-link").forEach((button) => button.classList.toggle("active", button.dataset.view === name));
  $("#current-section").textContent = titles[name] || "Overview";
  if (name === "records") loadRecords(1);
  window.scrollTo({ top: 0, behavior: "smooth" });
}
function recordFilters(page) {
  return {
    sro: $("#records-sro").value, type: $("#records-type").value,
    quarter: $("#records-quarter").value, q: $("#record-search").value.trim(),
    page: page, page_size: 10,
  };
}
async function loadRecords(page = state.recordsPage) {
  state.recordsPage = page;
  const body = $("#records-table-body");
  body.innerHTML = '<tr><td colspan="8" class="empty-cell">Loading records…</td></tr>';
  try {
    const result = await api("/api/records?" + queryString(recordFilters(page)));
    state.recordsPage = result.page;
    state.recordTotalPages = result.pages;
    $("#records-count").textContent = number(result.total) + " records";
    $("#page-label").textContent = result.page + " / " + result.pages;
    $("#page-prev").disabled = result.page <= 1;
    $("#page-next").disabled = result.page >= result.pages;
    if (!result.rows.length) { body.innerHTML = '<tr><td colspan="8" class="empty-cell">No records match your search.</td></tr>'; return; }
    body.innerHTML = result.rows.map((row) =>
      "<tr><td>" + escapeHTML(row.id) + "</td><td>" + escapeHTML(row.micro_market) + '<span class="table-sub">' + escapeHTML(row.sro_area) + "</span></td><td>" + escapeHTML(row.property_type) + "</td><td>" + escapeHTML(displayQuarter(row.quarter)) + "</td><td>" + number(row.area_sqft) + " sq ft</td><td>" + money(row.asking_price_inr, true) + "</td><td>" + money(row.declared_price_inr, true) + "</td><td>" + money(row.guidance_value_inr, true) + "</td></tr>"
    ).join("");
  } catch (error) { body.innerHTML = '<tr><td colspan="8" class="empty-cell">' + escapeHTML(error.message) + "</td></tr>"; }
}
function estimatePayload() {
  const market = selectedMarket();
  return {
    sro_area: $("#input-sro").value,
    property_type: $("#input-type").value,
    micro_market_id: market ? market.micro_market_id : "",
    micro_market: market ? market.micro_market : "New locality",
    as_of_quarter: $("#input-quarter").value,
    area_sqft: $("#input-area").value,
    bedrooms: $("#input-bedrooms").value,
    building_age_years: $("#input-age").value,
    floor_number: $("#input-floor").value,
    guideline_rate_inr_per_sqm: $("#input-rate").value,
    asking_price_inr: $("#input-asking").value,
    declared_price_inr: $("#input-declared").value,
  };
}
function renderEstimate(result) {
  const box = $("#estimate-result");
  const position = Math.max(3, Math.min(97, result.bound_position_percent));
  const comps = result.comparables.length ? result.comparables.map((row) =>
    '<div class="comparable-row"><strong>' + escapeHTML(row.micro_market) + "</strong><span>Ask " + money(row.asking_price_inr, true) + " · Deed " + money(row.declared_price_inr, true) + "</span><small>" + escapeHTML(row.property_type) + " · " + number(row.area_sqft) + " sq ft · " + escapeHTML(displayQuarter(row.quarter)) + "</small><small>" + row.similarity_percent + "% similar</small></div>"
  ).join("") : '<p class="result-disclaimer">No earlier records match this quarter.</p>';
  const lowerBound = Math.max(result.guidance_value_inr, result.declared_price_inr);
  const driverLabels = { log_area_sqft: "Area", bedrooms: "Bedrooms", building_age_years: "Age", floor_number: "Floor", property_type: "Property type", sro_area: "SRO area" };
  const drivers = Object.entries(result.hedonic_driver_contributions_log || {}).map(([key, value]) => {
    const effect = (Math.exp(Number(value)) - 1) * 100;
    return '<div class="driver-chip"><span>' + escapeHTML(driverLabels[key] || key) + '</span><strong>' + (effect > 0 ? '+' : '') + effect.toFixed(1) + '%</strong></div>';
  }).join("");
  box.innerHTML =
    '<div class="result-label">Bounded synthetic estimate</div><strong class="result-price">' + money(result.estimated_price_inr) + '</strong><div class="result-ppsf">' + money(result.price_per_sqft_inr) + " per sq ft</div>" +
    '<div class="result-range"><div class="range-head"><span>Nominal ' + result.calibration.nominal_coverage_percent + '% interval</span><strong>' + money(result.interval_lower_inr, true) + " – " + money(result.interval_upper_inr, true) + '</strong></div><div class="range-line"><span class="range-marker" style="left:' + position + '%"></span></div><div class="range-labels"><span>Lower bound ' + money(lowerBound, true) + "</span><span>Asking " + money(result.asking_price_inr, true) + "</span></div></div>" +
    '<div class="result-metrics"><div class="result-metric"><span>Guidance reference</span><strong>' + money(result.guidance_value_inr, true) + '</strong></div><div class="result-metric"><span>Declared signal</span><strong>' + money(result.declared_price_inr, true) + "</strong></div></div>" +
    '<div class="result-metrics"><div class="result-metric"><span>Learned asking markup</span><strong>' + (result.asking_markup_fraction * 100).toFixed(1) + '%</strong></div><div class="result-metric"><span>Under-declaration probability</span><strong>' + (result.underdeclaration_probability * 100).toFixed(1) + '%</strong></div><div class="result-metric"><span>Locality premium</span><strong>' + result.locality_premium_percent.toFixed(1) + '%</strong></div></div>' +
    '<div class="driver-heading">Hedonic price drivers</div><div class="driver-grid">' + drivers + '</div><small class="driver-note">Centered additive log-price contributions shown as percent equivalents; descriptive, not causal.</small>' +
    (result.locality_note ? '<p class="result-disclaimer">' + escapeHTML(result.locality_note) + '</p>' : '') +
    '<p class="result-disclaimer">Coverage was ' + result.calibration.observed_coverage_percent + '% on ' + number(result.calibration.interval_reference_rows) + ' generated anchors from ' + escapeHTML(displayQuarter(result.calibration.reference_quarter)) + '. This synthetic interval is not an appraisal or a real-market guarantee.</p>' +
    '<div class="comparable-heading">Earlier generated comparables</div><div class="comparable-list">' + comps + "</div>";
}
async function submitEstimate(event) {
  event.preventDefault();
  $("#estimate-error").textContent = "";
  const button = $(".estimate-submit");
  button.disabled = true;
  button.textContent = "Calculating…";
  try {
    const result = await api("/api/estimate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(estimatePayload()),
    });
    renderEstimate(result);
  } catch (error) { $("#estimate-error").textContent = error.message; }
  finally { button.disabled = false; button.innerHTML = 'Calculate estimate <span>→</span>'; }
}
function displayModelSnapshot() {
  const report = state.options.report;
  $("#model-snapshot").textContent =
    "BLP-Graph serving seed " + report.served_random_seed + ": spline hedonic head + market attention + prior-quarter GRU. Trained on " + number(report.training_rows) + " generated rows from 2023 Q2–Q3; conformal calibration on " + number(report.calibration_rows) + " flagged anchors from 2023 Q4; holdout: " + number(report.evaluation_rows) + " rows from 2024 Q1.";
  $("#about-records").textContent = number(report.records);
}
function wireInteractions() {
  $$(".nav-link").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
  $$("[data-go]").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.go)));
  ["#filter-sro", "#filter-type", "#filter-quarter"].forEach((selector) => $(selector).addEventListener("change", updateSummary));
  ["#records-sro", "#records-type", "#records-quarter"].forEach((selector) => $(selector).addEventListener("change", () => loadRecords(1)));
  $("#record-search").addEventListener("input", () => {
    clearTimeout(state.debounce);
    state.debounce = setTimeout(() => loadRecords(1), 180);
  });
  $("#page-prev").addEventListener("click", () => loadRecords(Math.max(1, state.recordsPage - 1)));
  $("#page-next").addEventListener("click", () => loadRecords(Math.min(state.recordTotalPages, state.recordsPage + 1)));
  $("#input-sro").addEventListener("change", () => syncMarketOptions(true));
  $("#input-type").addEventListener("change", () => syncMarketOptions(true));
  $("#input-market").addEventListener("change", () => selectedMarket() ? fillFromMarket() : fillSignalDefaults());
  $("#input-rate").addEventListener("input", updateSignalDefaults);
  $("#input-area").addEventListener("input", updateSignalDefaults);
  $("#estimate-form").addEventListener("submit", submitEstimate);
  window.addEventListener("resize", () => { if (state.overview) renderTrend(state.overview.trend); });
}
async function init() {
  wireInteractions();
  try {
    state.options = await api("/api/options");
    fillGlobalOptions();
    displayModelSnapshot();
    await updateSummary();
  } catch (error) {
    $("#overview-row-count").textContent = "The local app could not load its dataset.";
    $("#model-snapshot").textContent = error.message;
  }
}
init();
