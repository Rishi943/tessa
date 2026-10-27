import { defineConfig } from "vite";

export default defineConfig({
  build: { target: "es2022", chunkSizeWarningLimit: 800 },
  server: {
    // Ask Tessa runs in the Worker: `wrangler dev` on 8787 while vite serves the page
    proxy: { "/api": "http://127.0.0.1:8787" },
  },
});
