"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowRight, BookOpen, Check, Quotes } from "@phosphor-icons/react";
import { api, token } from "@/lib/api";
import "./Auth.css";

export default function Auth({ onDone }: { onDone: () => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [fields, setFields] = useState({ name: "", email: "", password: "" });
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  async function submit(event: React.FormEvent) {
    event.preventDefault(); setErr(""); setBusy(true);
    try {
      const response = await api(mode === "login" ? "/auth/login" : "/auth/register", { body: fields });
      if (!mounted.current) return;
      token.set(response.token); onDone();
    } catch (error: any) { if (mounted.current) setErr(error.message); } finally { if (mounted.current) setBusy(false); }
  }
  function changeMode() { setMode(mode === "login" ? "register" : "login"); setErr(""); }

  return (
    <main className="auth-page" id="main-content">
      <div className="auth-wordmark"><BookOpen size={32} weight="regular" aria-hidden="true" /><span>Research Assistant</span></div>
      <section className="auth-layout" aria-labelledby="auth-heading">
        <aside className="auth-introduction">
          <div className="auth-intro-copy"><span className="auth-intro-label">A workspace for your research</span><h2>From a stack of papers<br />to a clearer picture.</h2><p>Ask questions, compare findings, and follow every answer back to the evidence.</p></div>
          <div className="auth-source-note"><Quotes size={27} weight="regular" aria-hidden="true" /><p>Your sources stay close.<br /><span>Read the passage behind each citation.</span></p><div className="auth-source-line"><Check size={15} weight="bold" aria-hidden="true" /> Answers grounded in your selected papers</div></div>
        </aside>
        <div className="auth-form-panel">
          <header><h1 id="auth-heading">{mode === "login" ? "Welcome back" : "Create your workspace"}</h1><p>{mode === "login" ? "Sign in to continue your research." : "Start collecting papers and connecting ideas."}</p></header>
          <form onSubmit={submit} aria-busy={busy}>
            {mode === "register" && <label htmlFor="auth-name">Name<input id="auth-name" name="name" type="text" autoComplete="name" placeholder="Your name" value={fields.name} onChange={(event) => setFields({ ...fields, name: event.target.value })} disabled={busy} required /></label>}
            <label htmlFor="auth-email">Email<input id="auth-email" name="email" type="email" autoComplete="email" placeholder="you@university.edu" value={fields.email} onChange={(event) => setFields({ ...fields, email: event.target.value })} disabled={busy} required /></label>
            <label htmlFor="auth-password">Password<input id="auth-password" name="password" type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} minLength={8} placeholder="At least 8 characters" value={fields.password} onChange={(event) => setFields({ ...fields, password: event.target.value })} aria-describedby="auth-password-hint" disabled={busy} required /></label>
            <p className="auth-password-hint" id="auth-password-hint">Use 8 or more characters.</p>
            {err && <p className="err auth-error" role="alert">{err}</p>}
            <button className="btn auth-submit" disabled={busy} type="submit"><span>{busy ? (mode === "login" ? "Signing in…" : "Creating account…") : (mode === "login" ? "Sign in" : "Create account")}</span><ArrowRight size={19} aria-hidden="true" /></button>
          </form>
          <p className="auth-switch">{mode === "login" ? "New to Research Assistant?" : "Already have an account?"} <button type="button" onClick={changeMode} disabled={busy}>{mode === "login" ? "Create an account" : "Sign in"}</button></p>
          <p className="auth-privacy">Your library and uploaded papers are private to your account.</p>
        </div>
      </section>
    </main>
  );
}
