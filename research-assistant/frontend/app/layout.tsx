import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Research Assistant — literature review and evidence-backed gaps",
  description: "Upload papers, ask cited questions, compare studies, and find candidate research gaps with traceable evidence.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
