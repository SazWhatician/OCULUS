/**
 * OCULUS Sentinel API Client
 */

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
export const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export interface SessionItem {
  id: string;
  candidate_name: string;
  interviewer_name?: string;
  job_title: string;
  status: string;
  ingestion_mode: string;
  created_at: string;
  overall_risk_score?: number | null;
  risk_band?: "LOW" | "MODERATE" | "ELEVATED" | null;
}

export interface SessionDetail extends SessionItem {
  job_description?: string;
  custom_vocab?: string[];
  speaker_role_map?: Record<string, string>;
  baseline?: {
    session_id?: string;
    median_wpm?: number;
    baseline_wpm?: number;
    mad_wpm?: number;
    wpm_mad?: number;
    baseline_disfluency_rate?: number;
    disfluency_rate?: number;
    baseline_pause_ratio?: number;
  } | null;
  utterances?: any[];
  qa_evaluations?: any[];
  vision_events?: any[];
  report?: any;
}

export async function fetchSessions(): Promise<SessionItem[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/sessions`, { cache: "no-store" });
    if (!res.ok) throw new Error("Failed to fetch sessions");
    const data = await res.json();
    return data.sessions || [];
  } catch (err) {
    console.error("fetchSessions error:", err);
    return [];
  }
}

export async function fetchSession(id: string): Promise<SessionDetail | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/sessions/${id}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`Failed to fetch session ${id}`);
    return await res.json();
  } catch (err) {
    console.error("fetchSession error:", err);
    return null;
  }
}

export async function createSession(payload: {
  candidate_name: string;
  interviewer_name?: string;
  job_title: string;
  job_description?: string;
  ingestion_mode: "attendee_bot" | "browser_webrtc" | "demo_file";
  meeting_url?: string;
  bot_display_name?: string;
}) {
  const res = await fetch(`${API_BASE_URL}/api/v1/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Create session failed: ${res.statusText}`);
  return await res.json();
}

export async function uploadDemoFile(formData: FormData) {
  const res = await fetch(`${API_BASE_URL}/api/v1/demo/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`Demo upload failed: ${res.statusText}`);
  return await res.json();
}

export async function swapSpeakers(sessionId: string, speakerRoleMap: Record<string, string>) {
  const res = await fetch(`${API_BASE_URL}/api/v1/sessions/${sessionId}/speakers`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ speaker_role_map: speakerRoleMap }),
  });
  if (!res.ok) throw new Error("Failed to swap speakers");
  return await res.json();
}

export async function completeSession(sessionId: string) {
  const res = await fetch(`${API_BASE_URL}/api/v1/sessions/${sessionId}/complete`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Failed to complete session");
  return await res.json();
}

export async function fetchReport(sessionId: string) {
  const res = await fetch(`${API_BASE_URL}/api/v1/sessions/${sessionId}/report`, { cache: "no-store" });
  if (!res.ok) {
    // Try compat route
    const compatRes = await fetch(`${API_BASE_URL}/api/reports/${sessionId}`, { cache: "no-store" });
    if (!compatRes.ok) throw new Error("Report not yet generated or not found");
    return await compatRes.json();
  }
  return await res.json();
}
