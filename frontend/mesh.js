// Decorative face geometry. A static drawing keeps idle rendering cost low.
const meshCanvas = document.getElementById('face-mesh');
const mesh = meshCanvas.getContext('2d');
if (mesh) {
  const rows = [];
  for (let latitude = 1; latitude < 22; latitude++) {
    const pitch = (latitude / 22 - 0.5) * Math.PI;
    const height = Math.sin(pitch);
    const ring = [];
    for (let longitude = -12; longitude <= 12; longitude++) {
      const angle = longitude * Math.PI / 25;
      const x = Math.sin(angle) * Math.cos(pitch);
      const jaw = 1 - 0.27 * Math.max(height, 0);
      const nose = Math.exp(-40 * x * x - 12 * (height - 0.06) ** 2);
      const eyes = Math.exp(-42 * (Math.abs(x) - 0.39) ** 2 - 55 * (height + 0.2) ** 2);
      const depth = Math.cos(angle) * Math.cos(pitch) + 0.48 * nose - 0.2 * eyes;
      ring.push([230 + x * jaw * 86 + depth * 39, 136 + height * 115 - depth * 5]);
    }
    rows.push(ring);
  }
  mesh.lineWidth = 0.7;
  rows.slice(1).forEach((row, index) => {
    row.slice(1).forEach(([x, y], column) => {
      const upper = rows[index][column + 1];
      mesh.beginPath();
      mesh.moveTo(x, y);
      mesh.lineTo(...upper);
      mesh.lineTo(...row[column]);
      mesh.closePath();
      mesh.strokeStyle = column < 13 ? '#57dfcc90' : '#95baffb0';
      mesh.stroke();
    });
  });
}
