# 🎨 Legal AI Platform — Frontend Integration Guide

This directory contains the web user interface layer for the **Legal AI Intelligence & Negotiation Platform**.

The architecture is fully **decoupled**: the Python Backend (`backend/app_api.py`) runs as a REST API service on `http://localhost:8080`, allowing you to build your frontend using **React.js**, **Vue.js**, **Spring Boot**, or **Next.js**.

---

## 🔌 Available REST API Endpoints

| Method | Endpoint | Description | Request Payload | Response JSON |
| :---: | :--- | :--- | :--- | :--- |
| `GET` | `/api/status` | Server health check & LLM status | None | `{"status": "online", "llm_configured": true}` |
| `POST` | `/api/upload` | Upload PDF contract to analyze | `multipart/form-data` (`file`) | `{"status": "success", "doc_type": "...", "pages": 2}` |
| `GET` | `/api/summary` | Executive Summary & Classification | None | `{"classification": {...}, "summary": "..."}` |
| `GET` | `/api/heatmap` | 8-Category Clause Risk Heatmap | None | `[{"clause_title": "...", "risk_level": "🔴 HIGH", ...}]` |
| `POST` | `/api/negotiate` | Risk Analysis & Safer Clause Generator | `{"clause": "text..."}` | `{"risk_level": "...", "safer_alternative": "..."}` |
| `POST` | `/api/qa` | Grounded Q&A with Citation Cards | `{"question": "query..."}` | `{"answer": "...", "sources": [...]}` |
| `GET` | `/api/obligations` | Structured Duties & Deadlines | None | `[{"party": "...", "obligation": "...", "deadline_frequency": "..."}]` |
| `POST` | `/api/compare` | Compare an uploaded second version with the session's first contract | `multipart/form-data` (`file`) | `{"comparison_table": [...]}` |

Uploaded documents are isolated by an HTTP-only session cookie. Browser clients hosted on an allowed separate origin must send requests with `credentials: 'include'`. Configure additional development origins with `CORS_ALLOWED_ORIGINS`. The default maximum PDF size is 20 MB and can be changed with `MAX_UPLOAD_MB`.

When no LLM is configured (or `LEGAL_AI_OFFLINE=true`), the server does not invent legal conclusions: it returns cited deterministic extractions and marks risk/negotiation judgments as not assessed.

---

## ⚛️ React Integration Example (`src/api/legalApi.js`)

```javascript
const API_BASE = 'http://localhost:8080/api';

// 1. Upload Contract PDF
export async function uploadContract(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData, credentials: 'include' });
  return await res.json();
}

// 2. Fetch Clause Risk Heatmap
export async function getRiskHeatmap() {
  const res = await fetch(`${API_BASE}/heatmap`, { credentials: 'include' });
  return await res.json();
}

// 3. Evaluate Clause & Generate Safer Draft
export async function evaluateNegotiation(clauseText) {
  const res = await fetch(`${API_BASE}/negotiate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ clause: clauseText }),
    credentials: 'include',
  });
  return await res.json();
}

// 4. Grounded Legal Q&A
export async function askLegalQuestion(query) {
  const res = await fetch(`${API_BASE}/qa`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: query }),
    credentials: 'include',
  });
  return await res.json();
}
```

---

## 🍃 Spring Boot Controller Integration Example

```java
@RestController
@RequestMapping("/api/legal")
public class LegalAIClientController {

    private final RestTemplate restTemplate = new RestTemplate();
    private final String BACKEND_URL = "http://localhost:8080/api";

    @GetMapping("/heatmap")
    public ResponseEntity<String> getRiskHeatmap() {
        String response = restTemplate.getForObject(BACKEND_URL + "/heatmap", String.class);
        return ResponseEntity.ok(response);
    }
}
```
