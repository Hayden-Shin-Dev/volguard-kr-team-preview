// Quick Tunnels have no SSE. Poll the same saved quote and actual agent events.
(() => {
  class PreviewEventSource {
    constructor(url) {
      this.url = url;
      this.closed = false;
      this.opened = false;
      this.timer = setTimeout(() => this.poll(), 0);
    }
    async poll() {
      if (this.closed) return;
      try {
        const response = await fetch(this.url, { credentials: 'same-origin', cache: 'no-store' });
        if (!response.ok) throw new Error('연결 대기');
        const data = await response.json();
        if (this.closed) return;
        if (!this.opened) { this.opened = true; this.onopen?.({}); }
        this.onmessage?.({ data: JSON.stringify(data) });
      } catch (e) {
        if (!this.closed) { this.opened = false; this.onerror?.(e); }
      }
      if (!this.closed) this.timer = setTimeout(() => this.poll(), this.url.includes('/stream/') ? 5000 : 2500);
    }
    close() { this.closed = true; clearTimeout(this.timer); }
  }
  window.EventSource = PreviewEventSource;
  window.addEventListener('DOMContentLoaded', () => {
    const note = document.createElement('div');
    note.id = 'preview-notice';
    note.textContent = '팀 체험판 · 호스트 PC가 켜져 있을 때 접속 가능 · 보유정보는 브라우저별 분리 · AI 조사 하루 2회 / 전체 8회';
    const style = document.createElement('style');
    style.textContent = '#preview-notice{position:fixed;bottom:8px;left:50%;transform:translateX(-50%);z-index:9999;max-width:92vw;padding:8px 14px;border:1px solid #d8d5e8;border-radius:20px;background:#fffffff2;box-shadow:0 3px 20px #231d5b10;color:#55516b;font:11px/1.5 sans-serif;text-align:center;pointer-events:none}';
    document.head.append(style); document.body.append(note);
  });
})();
