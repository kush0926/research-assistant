import { Fragment, type ReactNode } from "react";
import "./Markdown.css";

/** A small Markdown renderer. Source content remains React text, never HTML. */
function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|__[^_]+__|`[^`]+`|\[[^\]]+\]\([^\s)]+\)|\[(?:P)?\d+\]|\*[^*\n]+\*)/g).map((token, i) => {
    if (/^(\*\*|__)/.test(token)) return <strong key={i}>{token.slice(2, -2)}</strong>;
    if (token.startsWith("`") && token.endsWith("`")) return <code key={i}>{token.slice(1, -1)}</code>;
    if (/^\[(?:P)?\d+\]$/.test(token)) return <span className="markdown-tag" key={i}>{token}</span>;
    const link = /^\[([^\]]+)\]\(([^\s)]+)\)$/.exec(token);
    if (link) {
      try {
        const url = new URL(link[2]);
        if (url.protocol === "https:" || url.protocol === "http:") return <a key={i} href={url.href} target="_blank" rel="noopener noreferrer">{link[1]}</a>;
      } catch { /* Invalid links stay visible as text. */ }
      return <Fragment key={i}>{token}</Fragment>;
    }
    if (token.startsWith("*") && token.endsWith("*") && token.length > 2) return <em key={i}>{token.slice(1, -1)}</em>;
    return <Fragment key={i}>{token}</Fragment>;
  });
}

export default function Markdown({ children, className = "" }: { children: string; className?: string }) {
  const lines = children.replace(/\r\n?/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  while (i < lines.length) {
    const text = lines[i].trim();
    if (!text) { i++; continue; }
    const key = i;
    if (text.startsWith("```")) {
      const code: string[] = []; i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) code.push(lines[i++]);
      if (i < lines.length) i++;
      blocks.push(<pre key={key}><code>{code.join("\n")}</code></pre>); continue;
    }
    const heading = /^(#{1,6})\s+(.+)$/.exec(text);
    if (heading) {
      const content = inline(heading[2]);
      blocks.push(heading[1].length <= 2 ? <h2 key={key}>{content}</h2> : <h3 key={key}>{content}</h3>); i++; continue;
    }
    if (/^([-*_])\1{2,}$/.test(text)) { blocks.push(<hr key={key} />); i++; continue; }
    if (/^>\s?/.test(text)) {
      const quote: string[] = [];
      while (i < lines.length && /^>\s?/.test(lines[i].trim())) quote.push(lines[i++].trim().replace(/^>\s?/, ""));
      blocks.push(<blockquote key={key}>{inline(quote.join(" "))}</blockquote>); continue;
    }
    if (/^\[P\d+\]\s+\S/.test(text)) {
      const references: ReactNode[] = [];
      while (i < lines.length && /^\[P\d+\]\s+\S/.test(lines[i].trim())) references.push(<li key={i}>{inline(lines[i++].trim())}</li>);
      blocks.push(<ul className="markdown-references" key={key}>{references}</ul>); continue;
    }
    const list = /^(?:[-*+]\s+|\d+[.)]\s+)/.exec(text);
    if (list) {
      const ordered = /^\d/.test(text); const items: ReactNode[] = [];
      const matcher = ordered ? /^\d+[.)]\s+(.+)$/ : /^[-*+]\s+(.+)$/;
      while (i < lines.length) {
        const match = matcher.exec(lines[i].trim());
        if (!match) break;
        let item = match[1]; i++;
        while (i < lines.length && /^\s+\S/.test(lines[i]) && !/^(?:[-*+]\s+|\d+[.)]\s+)/.test(lines[i].trim())) item += ` ${lines[i++].trim()}`;
        items.push(<li key={i}>{inline(item)}</li>);
      }
      blocks.push(ordered ? <ol key={key}>{items}</ol> : <ul key={key}>{items}</ul>); continue;
    }
    const paragraph: string[] = [text]; i++;
    while (i < lines.length && lines[i].trim() && !/^(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>|```|\[P\d+\]\s)/.test(lines[i].trim())) paragraph.push(lines[i++].trim());
    blocks.push(<p key={key}>{inline(paragraph.join(" "))}</p>);
  }
  return <div className={`research-markdown ${className}`.trim()}>{blocks}</div>;
}
