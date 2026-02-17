// graph.js — causal graph canvas renderer

function drawGraph(canvasId, demo) {
  const canvas = document.getElementById(canvasId);
  const parent = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = parent.offsetWidth * dpr;
  canvas.height = parent.offsetHeight * dpr;
  canvas.style.width = parent.offsetWidth + 'px';
  canvas.style.height = parent.offsetHeight + 'px';

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const w = parent.offsetWidth, h = parent.offsetHeight;

  const vars = demo.vars;
  const n = vars.length;
  const cx = w / 2, cy = h / 2;
  const radius = Math.min(w, h) * 0.33;

  // Position nodes in a circle
  const pos = {};
  vars.forEach((v, i) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * i) / n;
    pos[v] = { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
  });

  // Clear
  ctx.clearRect(0, 0, w, h);

  // Draw edges
  demo.edges.forEach(e => {
    const from = pos[e.s], to = pos[e.t];
    if (!from || !to) return;

    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.strokeStyle = e.dir ? '#111' : '#ccc';
    ctx.lineWidth = e.dir ? 1.2 : 1;
    ctx.setLineDash(e.dir ? [] : [4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Arrowhead
    if (e.dir) {
      const angle = Math.atan2(to.y - from.y, to.x - from.x);
      const hl = 7;
      const ax = to.x - 18 * Math.cos(angle);
      const ay = to.y - 18 * Math.sin(angle);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax - hl * Math.cos(angle - 0.4), ay - hl * Math.sin(angle - 0.4));
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax - hl * Math.cos(angle + 0.4), ay - hl * Math.sin(angle + 0.4));
      ctx.strokeStyle = '#111';
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
  });

  // Draw nodes
  vars.forEach(v => {
    const p = pos[v];

    // Circle
    ctx.beginPath();
    ctx.arc(p.x, p.y, 14, 0, Math.PI * 2);
    ctx.fillStyle = '#fbfbfb';
    ctx.fill();
    ctx.strokeStyle = '#111';
    ctx.lineWidth = 1.2;
    ctx.stroke();

    // Label
    const short = v.length <= 2;
    if (short) {
      ctx.fillStyle = '#111';
      ctx.font = '500 11px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(v, p.x, p.y);
    } else {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#111';
      ctx.fill();

      ctx.fillStyle = '#555';
      ctx.font = '400 9px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      const label = v.replace(/_/g, ' ');
      // Wrap long labels
      if (label.length > 12) {
        const words = label.split(' ');
        const mid = Math.ceil(words.length / 2);
        ctx.fillText(words.slice(0, mid).join(' '), p.x, p.y + 20);
        ctx.fillText(words.slice(mid).join(' '), p.x, p.y + 31);
      } else {
        ctx.fillText(label, p.x, p.y + 20);
      }
    }
  });
}
