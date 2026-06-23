import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright-core';
import { PNG } from 'pngjs';

const url = process.env.VERIFY_URL || 'http://127.0.0.1:8001/';
const chromePath =
  process.env.CHROME_PATH ||
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';

await mkdir('../tmp', { recursive: true });

const browser = await chromium.launch({
  executablePath: chromePath,
  headless: true,
  args: ['--use-angle=swiftshader', '--disable-dev-shm-usage'],
});

const viewports = [
  { name: 'desktop', width: 1365, height: 768 },
  { name: 'mobile', width: 390, height: 844 },
];

function getCanvasInfo() {
  const canvas = document.querySelector('canvas');
  if (!canvas) {
    return { canvasCount: 0, width: 0, height: 0 };
  }

  return { canvasCount: 1, width: canvas.width, height: canvas.height };
}

function inspectScreenshot(buffer) {
  const png = PNG.sync.read(buffer);
  const yStart = Math.floor(png.height * 0.32);
  const yEnd = Math.floor(png.height * 0.86);
  let brightPixels = 0;
  let hash = 2166136261;

  for (let y = yStart; y < yEnd; y += 3) {
    for (let x = 0; x < png.width; x += 3) {
      const index = (png.width * y + x) * 4;
      const r = png.data[index];
      const g = png.data[index + 1];
      const b = png.data[index + 2];
      const luminance = r + g + b;
      if (luminance > 150) brightPixels++;
      hash ^= r + (g << 8) + (b << 16);
      hash = Math.imul(hash, 16777619);
    }
  }

  return { width: png.width, height: png.height, brightPixels, hash: hash >>> 0 };
}

try {
  for (const viewport of viewports) {
    const page = await browser.newPage({ viewport });
    await page.goto(url, { waitUntil: 'networkidle' });
    await page.waitForSelector('canvas', { timeout: 15000 });
    await page.waitForTimeout(900);

    const canvas = await page.evaluate(getCanvasInfo);
    const firstShot = await page.screenshot();
    await page.waitForTimeout(500);
    const secondShot = await page.screenshot({ path: `../tmp/ui-${viewport.name}.png` });
    const first = inspectScreenshot(firstShot);
    const second = inspectScreenshot(secondShot);
    const layout = await page.evaluate(() => ({
      title: Boolean(document.body.innerText.includes('AI Portrait Studio')),
      cards: document.querySelectorAll('img[alt]').length,
      overflow: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    }));

    await page.close();

    if (!layout.title || layout.cards < 6) {
      throw new Error(`${viewport.name}: expected studio title and rendered image cards`);
    }
    if (layout.overflow > 2) {
      throw new Error(`${viewport.name}: horizontal overflow ${layout.overflow}px`);
    }
    if (canvas.canvasCount !== 1 || canvas.width < viewport.width || canvas.height < viewport.height) {
      throw new Error(`${viewport.name}: dotted canvas is not full viewport`);
    }
    if (first.brightPixels < 120) {
      throw new Error(`${viewport.name}: screenshot pixel check found too few bright dots`);
    }
    if (first.hash === second.hash) {
      throw new Error(`${viewport.name}: dotted canvas did not animate`);
    }

    console.log(
      `${viewport.name}: canvas ${canvas.width}x${canvas.height}, bright pixels ${first.brightPixels}->${second.brightPixels}, hash ${first.hash}->${second.hash}`,
    );
  }
} finally {
  await browser.close();
}
