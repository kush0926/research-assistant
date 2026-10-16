export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const token = {
  get: () => (typeof window === "undefined" ? "" : localStorage.getItem("ra_token") || ""),
  set: (t: string) => localStorage.setItem("ra_token", t),
  clear: () => localStorage.removeItem("ra_token"),
};

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) { super(message); this.status = status; }
}

export async function api(path: string, opts: { method?: string; body?: any; form?: FormData; signal?: AbortSignal } = {}) {
  const headers: Record<string, string> = {};
  const t = token.get();
  if (t) headers["Authorization"] = `Bearer ${t}`;
  let body: any = undefined;
  if (opts.form) body = opts.form;
  else if (opts.body !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(opts.body); }
  const res = await fetch(`${API}${path}`, { method: opts.method || (body ? "POST" : "GET"), headers, body, signal: opts.signal });
  if (!res.ok) {
    let msg = res.statusText;
    try { const j = await res.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new ApiError(res.status, msg);
  }
  return res.json();
}
