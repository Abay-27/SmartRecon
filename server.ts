console.log("Starting SmartRecon Server...");

import express from "express";
import cors from "cors";
import { createServer as createViteServer } from "vite";
import path from "path";
import { fileURLToPath } from "url";
import multer from "multer";
import * as XLSX from "xlsx";
import { GoogleGenAI } from "@google/genai";
import { parseISO, isValid, parse, format } from "date-fns";

const serverInstanceId = Math.random().toString(36).substring(2, 15);
let datasets: Record<string, any[]> = {};

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 100 * 1024 * 1024 } // 100MB
});

// Robust date parsing helper
function parseDate(dateStr: any): Date | null {
  if (dateStr instanceof Date) return dateStr;
  if (dateStr === null || dateStr === undefined || dateStr === "") return null;
  
  // Handle Excel serial dates (numbers)
  if (typeof dateStr === 'number') {
    return new Date((dateStr - 25569) * 86400 * 1000);
  }

  if (typeof dateStr !== 'string') return null;

  try {
    const isoDate = parseISO(dateStr);
    if (isValid(isoDate)) return isoDate;

    const formats = ["yyyy-MM-dd", "dd/MM/yyyy", "MM/dd/yyyy", "yyyy/MM/dd", "dd-MM-yyyy", "MMM dd, yyyy"];
    for (const fmt of formats) {
      const parsed = parse(dateStr, fmt, new Date());
      if (isValid(parsed)) return parsed;
    }
  } catch (e) {
    console.warn(`Failed to parse date: ${dateStr}`);
  }
  
  return null;
}

// Anomaly Detection Module - Enhanced with scoring and multi-variate analysis
function detectAnomalies(data: any[], mapping: any) {
  if (!data || data.length === 0) return [];

  const anomalies: any[] = [];
  const amounts = data.map(item => parseFloat(item[mapping.amount]) || 0);
  
  // Global stats
  const globalMean = amounts.reduce((a, b) => a + b, 0) / amounts.length;
  const globalStdDev = Math.sqrt(amounts.reduce((a, b) => a + Math.pow(b - globalMean, 2), 0) / amounts.length);

  // Customer-specific stats for "Personalized Normalcy"
  const customerStats: Record<string, { sum: number, count: number, amounts: number[], mean: number, stdDev: number }> = {};
  const customerVelocity: Record<string, Record<string, number>> = {}; // customer -> dateHour -> count
  
  data.forEach((item, index) => {
    const customer = mapping.customerName ? String(item[mapping.customerName] || "Unknown").toLowerCase() : "global";
    const amount = amounts[index];

    if (!customerStats[customer]) {
      customerStats[customer] = { sum: 0, count: 0, amounts: [], mean: 0, stdDev: 0 };
    }
    customerStats[customer].sum += amount;
    customerStats[customer].count += 1;
    customerStats[customer].amounts.push(amount);

    if (mapping.date) {
      const d = parseDate(item[mapping.date]);
      if (d) {
        const hourKey = `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}-${d.getHours()}`;
        if (!customerVelocity[customer]) customerVelocity[customer] = {};
        customerVelocity[customer][hourKey] = (customerVelocity[customer][hourKey] || 0) + 1;
      }
    }
  });

  // Finalize customer stats
  Object.keys(customerStats).forEach(customer => {
    const stats = customerStats[customer];
    stats.mean = stats.sum / stats.count;
    stats.stdDev = Math.sqrt(stats.amounts.reduce((a, b) => a + Math.pow(b - stats.mean, 2), 0) / stats.count);
  });

  data.forEach((item, index) => {
    let riskScore = 0;
    const reasons: string[] = [];
    const amount = amounts[index];
    const date = mapping.date ? parseDate(item[mapping.date]) : null;
    const customer = mapping.customerName ? String(item[mapping.customerName] || "Unknown").toLowerCase() : "global";
    const stats = customerStats[customer];

    // 1. Amount Anomaly (Global Z-Score)
    if (globalStdDev > 0) {
      const globalZScore = Math.abs((amount - globalMean) / globalStdDev);
      if (globalZScore > 3) {
        reasons.push(`Significant deviation from global average (Z-Score: ${globalZScore.toFixed(2)})`);
        riskScore += 0.4;
      }
    }

    // 2. Amount Anomaly (Personalized Z-Score) - Detects change in behavior
    if (stats.count > 5 && stats.stdDev > 0) {
      const personalZScore = Math.abs((amount - stats.mean) / stats.stdDev);
      if (personalZScore > 4) {
        reasons.push(`Unusual for this entity (Personal Z-Score: ${personalZScore.toFixed(2)})`);
        riskScore += 0.5;
      }
    }

    // 3. Time Anomaly (Odd hours: 11 PM - 5 AM)
    if (date) {
      const hours = date.getHours();
      if (hours >= 23 || hours <= 5) {
        reasons.push(`Transaction at suspicious hours (${hours}:00)`);
        riskScore += 0.3;
      }
      
      const day = date.getDay();
      if (day === 0 || day === 6) {
        reasons.push(`Weekend transaction (${day === 0 ? 'Sunday' : 'Saturday'})`);
        riskScore += 0.2;
      }
    }

    // 4. Velocity Anomalies (Clustering)
    if (date && customerVelocity[customer]) {
      const hourKey = `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}-${date.getHours()}`;
      const velocity = customerVelocity[customer][hourKey];
      if (velocity > 5) {
        reasons.push(`Rapid-fire burst detected (${velocity} txns in one hour)`);
        riskScore += 0.6;
      }
    }

    // 5. Fresh Entity (First time seen in this batch)
    if (stats.count === 1) {
      // Not necessarily an anomaly, but adds to risk if combined with others
      // reasons.push("New entity detected");
      riskScore += 0.1;
    }

    if (riskScore >= 0.5 || reasons.length >= 2) {
      anomalies.push({
        id: `anom-${Math.random().toString(36).substring(7)}`,
        record: item,
        reasons,
        riskScore: Math.min(riskScore, 1.0),
        severity: riskScore > 0.8 ? "critical" : (riskScore > 0.6 ? "high" : "medium"),
        amount,
        customer: item[mapping.customerName] || "Unknown",
        timestamp: date ? date.toISOString() : new Date().toISOString()
      });
    }
  });

  return anomalies;
}

async function startServer() {
  console.log(">>> Initializing SmartRecon Server...");
  const app = express();
  const PORT = 3000;

  // 1. Basic configuration
  app.use(cors());
  
  // 2. Immediate Health check
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok", timestamp: new Date().toISOString(), instanceId: serverInstanceId });
  });

  app.get("/api/ping", (req, res) => {
    res.json({ pong: true, time: new Date().toISOString(), instanceId: serverInstanceId, env: process.env.NODE_ENV });
  });

  // 3. Start listening EARLY so API is reachable while Vite is booting
  const server = app.listen(PORT, "0.0.0.0", () => {
    console.log(`>>> API Layer active at http://0.0.0.0:${PORT}`);
    console.log(`>>> [${new Date().toISOString()}] Instance ID: ${serverInstanceId}`);
  });

  // 4. Global logging middleware
  app.use((req, res, next) => {
    if (req.url.startsWith("/api/")) {
      console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
    }
    next();
  });

  // 5. Upload route
  app.post("/api/upload", (req, res) => {
    console.log(`[${new Date().toISOString()}] Incoming upload request...`);
    upload.single("file")(req, res, (err) => {
      if (err) {
        console.error("Multer Error:", err);
        return res.status(400).json({ 
          error: err.message || "File upload failed",
          code: (err as any).code 
        });
      }

      if (!req.file) {
        console.warn("Upload request with no file");
        return res.status(400).json({ error: "No file uploaded" });
      }

      try {
        console.log(`Processing file: ${req.file.originalname} (${req.file.size} bytes)`);
        const workbook = XLSX.read(req.file.buffer, { type: "buffer" });
        const sheetName = workbook.SheetNames[0];
        const data = XLSX.utils.sheet_to_json(workbook.Sheets[sheetName]);
        
        if (!data || data.length === 0) {
          return res.status(400).json({ error: "File is empty or invalid" });
        }

        const id = Math.random().toString(36).substring(7);
        datasets[id] = data;
        
        console.log(`File processed successfully. ID: ${id}, Rows: ${data.length}`);
        res.json({ id, headers: Object.keys(data[0] || {}), preview: data.slice(0, 5) });
      } catch (error) {
        console.error("Upload processing error:", error);
        res.status(500).json({ error: "Failed to parse file: " + (error instanceof Error ? error.message : String(error)) });
      }
    });
  });

  app.use(express.json({ limit: "100mb" }));
  app.use(express.urlencoded({ limit: "100mb", extended: true }));

  // 6. Other API Routes
  app.post("/api/reconcile", (req, res) => {
    const { sourceAId, sourceBId, mappingA, mappingB, rules } = req.body;
    console.log(`[${new Date().toISOString()}] Reconciling ${sourceAId} and ${sourceBId}`);
    
    const dataA = datasets[sourceAId];
    const dataB = datasets[sourceBId];

    if (!dataA || !dataB) return res.status(400).json({ error: "Datasets not found" });

    try {
      const results = performReconciliation(dataA, dataB, mappingA, mappingB, rules);
      res.json(results);
    } catch (error) {
      console.error("Reconciliation error:", error);
      res.status(500).json({ error: "Reconciliation engine failed" });
    }
  });

  // Catch-all for /api to prevent falling through to SPA fallback
  app.all("/api/*", (req, res) => {
    console.warn(`[${new Date().toISOString()}] 404 API Route: ${req.method} ${req.url}`);
    res.status(404).json({ error: `API route ${req.method} ${req.url} not found` });
  });

  // 7. Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    console.log(">>> Booting Vite middleware (this may take a few seconds)...");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
    console.log(">>> Vite middleware READY");
  } else {
    console.log(">>> Using Static middleware (Production)");
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  // 8. Global Error Handler
  app.use((err: any, req: any, res: any, next: any) => {
    console.error("!!! Express Error:", err);
    res.status(err.status || 500).json({ 
      error: err.message || "Internal Server Error",
      path: req.path
    });
  });
}

function performReconciliation(dataA: any[], dataB: any[], mapA: any, mapB: any, rules: any) {
  const matched: any[] = [];
  const unmatchedA: any[] = [];
  let unmatchedB: any[] = [...dataB];
  const mismatches: any[] = [];

  const { 
    amountTolerance = 0, 
    dateTolerance = 0, 
    nameSimilarityThreshold = 0.8,
    algorithm = "exact" 
  } = rules;

  // Detect Anomalies first
  const anomaliesA = detectAnomalies(dataA, mapA);
  const anomaliesB = detectAnomalies(dataB, mapB);

  // Optimization: Pre-index Source B for exact matches (ID + Amount)
  const indexB = new Map<string, any[]>();
  unmatchedB.forEach(itemB => {
    const idB = String(itemB[mapB.transactionId] || "");
    const amountB = String(itemB[mapB.amount] || "0");
    const key = `${idB}_${amountB}`;
    if (!indexB.has(key)) indexB.set(key, []);
    indexB.get(key)!.push(itemB);
  });

  const remainingB = new Set(unmatchedB);

  dataA.forEach((itemA) => {
    const idA = String(itemA[mapA.transactionId] || "");
    const amountA = String(itemA[mapA.amount] || "0");
    const key = `${idA}_${amountA}`;

    const matchesB = indexB.get(key);
    if (matchesB && matchesB.length > 0) {
      const matchB = matchesB.shift()!;
      matched.push({ sourceA: itemA, sourceB: matchB, type: "exact" });
      remainingB.delete(matchB);
    } else {
      unmatchedA.push(itemA);
    }
  });

  unmatchedB = Array.from(remainingB);

  return {
    summary: {
      totalA: dataA.length,
      totalB: dataB.length,
      matched: matched.length,
      mismatches: mismatches.length,
      unmatchedA: unmatchedA.length,
      unmatchedB: unmatchedB.length,
      anomaliesA: anomaliesA.length,
      anomaliesB: anomaliesB.length,
      matchedPercent: ((matched.length + mismatches.length) / dataA.length * 100).toFixed(2),
      variance: mismatches.reduce((acc, m) => acc + m.diff.amount, 0).toFixed(2)
    },
    matched,
    unmatchedA,
    unmatchedB,
    mismatches,
    anomaliesA,
    anomaliesB
  };
}

startServer().catch(err => {
  console.error("!!! Failed to start server:", err);
});

// Process-level error handling
process.on("uncaughtException", (err) => {
  console.error("!!! Uncaught Exception:", err);
});

process.on("unhandledRejection", (reason, promise) => {
  console.error("!!! Unhandled Rejection at:", promise, "reason:", reason);
});
