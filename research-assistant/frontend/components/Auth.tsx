"use client";
import { useState } from "react";
import { api, token } from "@/lib/api";

export default function Auth({ onDone }: { onDone: () => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [f, setF] = useState({ name: "", email: "", password: "" });
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  async function submit(e: React.FormEvent) {
    e.preventDefault(); setErr(""); setBusy(true);
    try {
      const r = await api(mode === "login" ? "/auth/login" : "/auth/register", { body: f });
      token.set(r.token); onDone();
    } catch (x: any) { setErr(x.message); } finally { setBusy(false); }
  }
  return (
    <main style={{ maxWidth: 400, margin: "10vh auto", padding: "0 1rem" }}>
      <h1>Research Assistant</h1>
      <p className="muted">Read fewer papers, trust more of what you read. Every answer links to its source passage.</p>
      <form className="card" onSubmit={submit}>
        {mode === "register" && <p><label>Name<input type="text" value={f.name} onChange={(e) => setF({ ...f, name: e.target.value })} required /></label></p>}
        <p><label>Email<input type="email" value={f.email} onChange={(e) => setF({ ...f, email: e.target.value })} required /></label></p>
        <p><label>Password (8+ characters)<input type="password" minLength={8} value={f.password} onChange={(e) => setF({ ...f, password: e.target.value })} required /></label></p>
        {err && <p className="err">{err}</p>}
        <button className="btn" disabled={busy}>{mode === "login" ? "Sign in" : "Create account"}</button>{" "}
        <button type="button" className="btn ghost" onClick={() => setMode(mode === "login" ? "register" : "login")}>
          {mode === "login" ? "Create an account" : "I have an account"}
        </button>
      </form>
    </main>
  );
}
