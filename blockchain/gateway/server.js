"use strict";

const fs = require("fs");
const path = require("path");
const express = require("express");
const { Gateway, Wallets } = require("fabric-network");

const app = express();
app.use(express.json());

const CHANNEL = "tourismchannel";
const CHAINCODE = "tourist-safety";
const PORT = process.env.GATEWAY_PORT || 7059;

const ccpPath = path.resolve(
  __dirname,
  "../../fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/connection-org1.json"
);

const mspPath = path.resolve(
  __dirname,
  "../../fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
);

async function buildWallet() {
  const cert = fs.readFileSync(path.join(mspPath, "signcerts", "cert.pem")).toString();
  const keyDir = path.join(mspPath, "keystore");
  const keyFile = fs.readdirSync(keyDir)[0];
  const key = fs.readFileSync(path.join(keyDir, keyFile)).toString();

  const wallet = await Wallets.newInMemoryWallet();
  await wallet.put("org1Admin", {
    credentials: { certificate: cert, privateKey: key },
    mspId: "Org1MSP",
    type: "X.509"
  });
  return wallet;
}

async function getContract() {
  if (!fs.existsSync(ccpPath)) {
    throw new Error(`Connection profile not found: ${ccpPath}`);
  }
  if (!fs.existsSync(mspPath)) {
    throw new Error(`MSP path not found: ${mspPath}`);
  }

  const ccp = JSON.parse(fs.readFileSync(ccpPath, "utf8"));
  const wallet = await buildWallet();

  const gateway = new Gateway();
  await gateway.connect(ccp, {
    wallet,
    identity: "org1Admin",
    discovery: { enabled: true, asLocalhost: true }
  });

  const network = await gateway.getNetwork(CHANNEL);
  const contract = network.getContract(CHAINCODE);

  return { gateway, contract };
}

app.get("/health", (_req, res) => {
  res.json({ ok: true, channel: CHANNEL, chaincode: CHAINCODE });
});

app.get("/fabric/did/:did", async (req, res) => {
  try {
    const { did } = req.params;
    const { gateway, contract } = await getContract();
    const result = await contract.evaluateTransaction("GetDID", did);
    await gateway.disconnect();
    const raw = result.toString();
    let parsed = raw;
    try {
      parsed = JSON.parse(raw);
    } catch (_) {
      // keep raw string
    }
    res.json({ result: parsed });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get("/fabric/link/:sessionId", async (req, res) => {
  try {
    const { sessionId } = req.params;
    const { gateway, contract } = await getContract();
    const result = await contract.evaluateTransaction("GetLinkBySession", sessionId);
    await gateway.disconnect();
    const raw = result.toString();
    let parsed = raw;
    try {
      parsed = JSON.parse(raw);
    } catch (_) {
      // keep raw string
    }
    res.json({ result: parsed });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/fabric/register", async (req, res) => {
  try {
    const { did, piiHash, faceHash, timestamp } = req.body;
    const { gateway, contract } = await getContract();
    const tx = contract.createTransaction("RegisterDID");
    const txId = tx.getTransactionId();
    const result = await tx.submit(did, piiHash, faceHash || "", timestamp || "");
    await gateway.disconnect();
    res.json({ txId, result: result.toString() });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/fabric/link", async (req, res) => {
  try {
    const { sessionId, did, confidence, timestamp } = req.body;
    const { gateway, contract } = await getContract();
    const tx = contract.createTransaction("LinkTrackingSession");
    const txId = tx.getTransactionId();
    const result = await tx.submit(sessionId, did, String(confidence), timestamp || "");
    await gateway.disconnect();
    res.json({ txId, result: result.toString() });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post("/fabric/incident", async (req, res) => {
  try {
    const { incidentId, did, sessionId, evidenceHash, timestamp, notes } = req.body;
    const { gateway, contract } = await getContract();
    const tx = contract.createTransaction("LogIncident");
    const txId = tx.getTransactionId();
    const result = await tx.submit(
      incidentId,
      did,
      sessionId || "",
      evidenceHash,
      timestamp || "",
      notes || ""
    );
    await gateway.disconnect();
    res.json({ txId, result: result.toString() });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Fabric Gateway running on http://localhost:${PORT}`);
});
