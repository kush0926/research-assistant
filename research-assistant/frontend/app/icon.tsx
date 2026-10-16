import { ImageResponse } from "next/og";
import { BookOpen } from "@phosphor-icons/react/dist/ssr";

export const size = { width: 64, height: 64 };
export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    <div style={{ width: "100%", height: "100%", background: "#f2f7f6", borderRadius: 14, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <BookOpen size={44} color="#0e6b66" weight="regular" />
    </div>,
    size,
  );
}
