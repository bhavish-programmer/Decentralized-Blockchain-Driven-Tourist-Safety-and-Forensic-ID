"use strict";

const { Contract } = require("fabric-contract-api");

class TouristSafetyContract extends Contract {
  async RegisterDID(ctx, did, piiHash, faceHash, timestamp) {
    this._requireOrg(ctx, ["Org1MSP"], "RegisterDID");
    this._assertNonEmpty(did, "did");
    this._assertNonEmpty(piiHash, "piiHash");

    const didKey = ctx.stub.createCompositeKey("DID", [did]);
    const existing = await this._getStateAsJSON(ctx, didKey);
    if (existing) {
      throw new Error(`DID already exists: ${did}`);
    }

    const record = {
      type: "DID",
      did,
      piiHash,
      faceHash: faceHash || null,
      createdAt: timestamp || this._nowISO(ctx),
      createdBy: ctx.clientIdentity.getMSPID(),
      txId: ctx.stub.getTxID()
    };

    await this._putStateFromJSON(ctx, didKey, record);
    return JSON.stringify(record);
  }

  async LinkTrackingSession(ctx, sessionId, did, confidence, timestamp) {
    this._requireOrg(ctx, ["Org1MSP"], "LinkTrackingSession");
    this._assertNonEmpty(sessionId, "sessionId");
    this._assertNonEmpty(did, "did");

    const didKey = ctx.stub.createCompositeKey("DID", [did]);
    const didRecord = await this._getStateAsJSON(ctx, didKey);
    if (!didRecord) {
      throw new Error(`DID not found: ${did}`);
    }

    const linkKey = ctx.stub.createCompositeKey("LINK", [sessionId]);
    const existing = await this._getStateAsJSON(ctx, linkKey);
    if (existing) {
      throw new Error(`Tracking session already linked: ${sessionId}`);
    }

    const conf = Number.parseFloat(confidence);
    if (Number.isNaN(conf)) {
      throw new Error("confidence must be a number");
    }

    const record = {
      type: "LINK",
      sessionId,
      did,
      confidence: conf,
      linkedAt: timestamp || this._nowISO(ctx),
      linkedBy: ctx.clientIdentity.getMSPID(),
      txId: ctx.stub.getTxID()
    };

    await this._putStateFromJSON(ctx, linkKey, record);
    return JSON.stringify(record);
  }

  async LogIncident(ctx, incidentId, did, sessionId, evidenceHash, timestamp, notes) {
    this._requireOrg(ctx, ["Org2MSP", "Org3MSP"], "LogIncident");
    this._assertNonEmpty(incidentId, "incidentId");
    this._assertNonEmpty(did, "did");
    this._assertNonEmpty(evidenceHash, "evidenceHash");

    const didKey = ctx.stub.createCompositeKey("DID", [did]);
    const didRecord = await this._getStateAsJSON(ctx, didKey);
    if (!didRecord) {
      throw new Error(`DID not found: ${did}`);
    }

    const incidentKey = ctx.stub.createCompositeKey("INC", [did, incidentId]);
    const existing = await this._getStateAsJSON(ctx, incidentKey);
    if (existing) {
      throw new Error(`Incident already exists: ${incidentId}`);
    }

    const record = {
      type: "INCIDENT",
      incidentId,
      did,
      sessionId: sessionId || null,
      evidenceHash,
      timestamp: timestamp || this._nowISO(ctx),
      notes: notes || null,
      reportedBy: ctx.clientIdentity.getMSPID(),
      txId: ctx.stub.getTxID()
    };

    await this._putStateFromJSON(ctx, incidentKey, record);
    return JSON.stringify(record);
  }

  async GetDID(ctx, did) {
    this._assertNonEmpty(did, "did");
    const didKey = ctx.stub.createCompositeKey("DID", [did]);
    const record = await this._getStateAsJSON(ctx, didKey);
    if (!record) {
      throw new Error(`DID not found: ${did}`);
    }
    return JSON.stringify(record);
  }

  async GetLinkBySession(ctx, sessionId) {
    this._assertNonEmpty(sessionId, "sessionId");
    const linkKey = ctx.stub.createCompositeKey("LINK", [sessionId]);
    const record = await this._getStateAsJSON(ctx, linkKey);
    if (!record) {
      throw new Error(`Link not found for session: ${sessionId}`);
    }
    return JSON.stringify(record);
  }

  async GetIncident(ctx, did, incidentId) {
    this._assertNonEmpty(did, "did");
    this._assertNonEmpty(incidentId, "incidentId");
    const incidentKey = ctx.stub.createCompositeKey("INC", [did, incidentId]);
    const record = await this._getStateAsJSON(ctx, incidentKey);
    if (!record) {
      throw new Error(`Incident not found: ${incidentId}`);
    }
    return JSON.stringify(record);
  }

  async GetIncidentsByDID(ctx, did) {
    this._assertNonEmpty(did, "did");
    const iterator = await ctx.stub.getStateByPartialCompositeKey("INC", [did]);
    const results = [];

    while (true) {
      const res = await iterator.next();
      if (res.value && res.value.value) {
        const raw = res.value.value.toString("utf8");
        results.push(JSON.parse(raw));
      }
      if (res.done) {
        await iterator.close();
        break;
      }
    }

    return JSON.stringify(results);
  }

  _requireOrg(ctx, allowed, action) {
    const msp = ctx.clientIdentity.getMSPID();
    if (!allowed.includes(msp)) {
      throw new Error(`${action} not permitted for MSP ${msp}`);
    }
  }

  _assertNonEmpty(value, name) {
    if (value === undefined || value === null || `${value}`.trim() === "") {
      throw new Error(`${name} is required`);
    }
  }

  _nowISO(ctx) {
    const ts = ctx.stub.getTxTimestamp();
    const seconds = ts.seconds && typeof ts.seconds.toNumber === "function"
      ? ts.seconds.toNumber()
      : Number(ts.seconds);
    const nanos = ts.nanos ? Number(ts.nanos) : 0;
    const millis = seconds * 1000 + Math.floor(nanos / 1e6);
    return new Date(millis).toISOString();
  }

  async _getStateAsJSON(ctx, key) {
    const data = await ctx.stub.getState(key);
    if (!data || data.length === 0) return null;
    return JSON.parse(data.toString());
  }

  async _putStateFromJSON(ctx, key, value) {
    const buffer = Buffer.from(JSON.stringify(value));
    await ctx.stub.putState(key, buffer);
  }
}

module.exports = TouristSafetyContract;
