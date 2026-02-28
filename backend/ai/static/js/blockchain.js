"use strict";

async function fetchLedger(limit = 20) {
    const response = await fetch(`/api/blockchain/ledger?limit=${limit}`);
    if (!response.ok) {
        throw new Error(`Ledger fetch failed (${response.status})`);
    }
    return response.json();
}

function formatTimestamp(value) {
    if (!value) return "—";
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return value;
    return d.toISOString().replace("T", " ").replace("Z", " UTC");
}

function renderLedgerRows(entries) {
    const tbody = document.getElementById("ledgerBody");
    if (!tbody) return;

    if (!entries || entries.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="ledger-empty">No ledger records found yet.</td></tr>';
        return;
    }

    tbody.innerHTML = entries.map(entry => {
        const typeClass = entry.type ? entry.type.toLowerCase() : "did";
        const tagLabel = entry.type || "DID";
        const confidence = entry.confidence != null ? Number(entry.confidence).toFixed(3) : "—";
        return `
        <tr>
            <td><span class="tag ${typeClass}">${tagLabel}</span></td>
            <td class="mono">${entry.recordId || "—"}</td>
            <td class="mono">${entry.did || "—"}</td>
            <td>${confidence}</td>
            <td>${formatTimestamp(entry.timestamp)}</td>
            <td>${entry.org || "—"}</td>
            <td class="mono">${entry.txId || "—"}</td>
        </tr>`;
    }).join("");
}

function setStatus(text, ok = true) {
    const badge = document.getElementById("ledgerStatus");
    if (!badge) return;
    badge.textContent = text;
    badge.style.color = ok ? "#32f6ff" : "#ff8674";
    badge.style.borderColor = ok ? "rgba(50, 246, 255, 0.4)" : "rgba(255, 134, 116, 0.5)";
}

function updateHeader(meta) {
    const channel = document.getElementById("bcChannel");
    const chaincode = document.getElementById("bcChaincode");
    if (channel) channel.textContent = meta?.channel || "tourismchannel";
    if (chaincode) chaincode.textContent = meta?.chaincode || "tourist-safety";
}

async function loadLedger() {
    setStatus("Loading…", true);
    try {
        const data = await fetchLedger(20);
        updateHeader(data.gateway);

        if (!data.fabricEnabled) {
            setStatus("Fabric disabled (ENABLE_FABRIC=false)", false);
        } else if (data.gatewayOk) {
            setStatus("Gateway online", true);
        } else {
            setStatus("Gateway unreachable", false);
        }

        const entries = [...(data.didRecords || []), ...(data.linkRecords || [])];
        entries.sort((a, b) => new Date(b.timestamp || 0) - new Date(a.timestamp || 0));
        renderLedgerRows(entries);
    } catch (err) {
        setStatus("Failed to load ledger", false);
        renderLedgerRows([]);
        console.error(err);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const refresh = document.getElementById("refreshLedger");
    if (refresh) {
        refresh.addEventListener("click", loadLedger);
    }
    loadLedger();
});
