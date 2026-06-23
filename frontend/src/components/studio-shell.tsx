import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Check,
  Cloud,
  Cpu,
  Image as ImageIcon,
  Lock,
  Moon,
  Save,
  Sparkles,
  Sun,
  Upload,
  X,
} from 'lucide-react';
import { useTheme } from 'next-themes';
import { DottedSurface } from '@/components/dotted-surface';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';

type Mode = 'demo' | 'diffusion' | 'api';
type Decision = 'accepted' | 'rejected' | '';

type Health = {
  device: string;
  cuda_available: boolean;
  gpu_name: string;
  api_key_configured: boolean;
  api_image_model: string;
};

type Variation = {
  id: number;
  label: string;
  image_url: string;
  seed: number;
  strength: number;
  provider?: string;
  face_preservation: boolean;
  detected_faces?: number;
};

type DecisionState = Record<number, { decision: Decision; reason: string }>;

const modes: Array<{ value: Mode; label: string; detail: string; icon: typeof Sparkles }> = [
  { value: 'demo', label: 'Preview', detail: 'Fast interface pass', icon: Sparkles },
  { value: 'api', label: 'API studio', detail: 'Production edits', icon: Cloud },
  { value: 'diffusion', label: 'GPU studio', detail: 'Local CUDA run', icon: Cpu },
];

const sampleVariations: Variation[] = [
  {
    id: 1,
    label: 'Natural lighting',
    provider: 'preview',
    image_url: '/examples/session_001/variation_1.png',
    seed: 4201,
    strength: 0.36,
    face_preservation: true,
    detected_faces: 1,
  },
  {
    id: 2,
    label: 'Cinematic tint',
    provider: 'preview',
    image_url: '/examples/session_001/variation_2.png',
    seed: 4202,
    strength: 0.42,
    face_preservation: true,
    detected_faces: 1,
  },
  {
    id: 3,
    label: 'Studio headshot',
    provider: 'preview',
    image_url: '/examples/session_001/variation_3.png',
    seed: 4203,
    strength: 0.34,
    face_preservation: true,
    detected_faces: 1,
  },
  {
    id: 4,
    label: 'Editorial polish',
    provider: 'preview',
    image_url: '/examples/session_001/variation_4.png',
    seed: 4204,
    strength: 0.44,
    face_preservation: true,
    detected_faces: 1,
  },
  {
    id: 5,
    label: 'Soft luxury retouch',
    provider: 'preview',
    image_url: '/examples/session_001/variation_5.png',
    seed: 4205,
    strength: 0.33,
    face_preservation: true,
    detected_faces: 1,
  },
];

function emptyDecisions(variations: Variation[]): DecisionState {
  return variations.reduce<DecisionState>((acc, item) => {
    acc[item.id] = { decision: '', reason: '' };
    return acc;
  }, {});
}

export function StudioShell() {
  const { resolvedTheme, setTheme } = useTheme();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [mode, setMode] = useState<Mode>('demo');
  const [preserveFaces, setPreserveFaces] = useState(true);
  const [seed, setSeed] = useState(4200);
  const [file, setFile] = useState<File | null>(null);
  const [inputPreview, setInputPreview] = useState('/examples/session_001/input.png');
  const [inputState, setInputState] = useState('Sample loaded');
  const [sessionId, setSessionId] = useState('');
  const [variations, setVariations] = useState<Variation[]>(sampleVariations);
  const [decisions, setDecisions] = useState<DecisionState>(emptyDecisions(sampleVariations));
  const [status, setStatus] = useState('Ready');
  const [summary, setSummary] = useState(
    'Use Preview to test the workflow. Switch to API studio when the local model cannot create enough separation between the five professional looks.',
  );
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    fetch('/api/health')
      .then((response) => response.json())
      .then((data: Health) => setHealth(data))
      .catch(() => setHealth(null));
  }, []);

  const selectedCount = useMemo(
    () => Object.values(decisions).filter((item) => item.decision).length,
    [decisions],
  );

  const canSave = Boolean(sessionId) && selectedCount === variations.length && variations.length > 0;
  const activeTheme = resolvedTheme === 'light' ? 'light' : 'dark';

  function handleFile(nextFile: File | null) {
    if (!nextFile) return;
    setFile(nextFile);
    setInputPreview(URL.createObjectURL(nextFile));
    setInputState(nextFile.name);
    setSummary('Portrait loaded. Choose a studio mode and create the five-look set.');
  }

  async function handleGenerate() {
    if (!file) {
      setStatus('Needs input');
      setSummary('Upload a portrait before creating a studio set.');
      fileInputRef.current?.click();
      return;
    }

    const form = new FormData();
    form.append('image', file);
    form.append('mode', mode);
    form.append('base_seed', String(seed));
    form.append('preserve_faces', String(preserveFaces));

    setIsGenerating(true);
    setStatus('Processing');
    setSummary('Rendering five professional portrait looks...');

    try {
      const response = await fetch('/api/generate', { method: 'POST', body: form });
      const data = await response.json();

      if (!response.ok) {
        setStatus('Needs attention');
        setSummary(data.detail || 'Generation failed.');
        return;
      }

      setSessionId(data.session_id);
      setInputPreview(data.input_image_url);
      setInputState('Session image');
      setVariations(data.variations);
      setDecisions(emptyDecisions(data.variations));
      setStatus('Review');
      setSummary('Review all five looks. Keep the outputs that feel production-ready and pass the rest.');
    } catch {
      setStatus('Needs attention');
      setSummary('The studio service is not reachable.');
    } finally {
      setIsGenerating(false);
    }
  }

  async function handleSave() {
    if (!canSave) {
      setSummary('Every studio look needs a keep/pass decision before saving.');
      return;
    }

    setIsSaving(true);
    setStatus('Saving');
    const payload = {
      decisions: variations.map((item) => ({
        variation_id: item.id,
        decision: decisions[item.id].decision,
        reason: decisions[item.id].reason,
      })),
    };

    try {
      const response = await fetch(`/api/sessions/${sessionId}/decisions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      setStatus(response.ok ? 'Saved' : 'Needs attention');
      setSummary(response.ok ? data.summary : data.detail || 'Could not save the decisions.');
    } catch {
      setStatus('Needs attention');
      setSummary('The decision service is not reachable.');
    } finally {
      setIsSaving(false);
    }
  }

  function setDecision(id: number, decision: Decision) {
    setDecisions((current) => ({
      ...current,
      [id]: { ...(current[id] || { reason: '' }), decision },
    }));
  }

  function setReason(id: number, reason: string) {
    setDecisions((current) => ({
      ...current,
      [id]: { ...(current[id] || { decision: '' }), reason },
    }));
  }

  return (
    <div className="relative min-h-screen overflow-hidden bg-background text-foreground scanline">
      <DottedSurface />
      <div className="pointer-events-none fixed inset-0 z-[1] bg-[radial-gradient(circle_at_50%_34%,transparent_0,transparent_16rem,hsl(var(--background)/0.38)_31rem,hsl(var(--background)/0.92)_100%)]" />
      <div className="pointer-events-none fixed inset-x-0 top-0 z-[2] h-40 bg-gradient-to-b from-background via-background/82 to-transparent" />
      <div className="pointer-events-none fixed inset-x-0 bottom-0 z-[2] h-44 bg-gradient-to-t from-background via-background/80 to-transparent" />

      <header className="fixed inset-x-0 top-0 z-20 flex items-center justify-between px-5 py-4 md:px-8">
        <div className="flex items-center gap-3">
          <div className="grid h-8 w-8 place-items-center rounded-md border border-border bg-card/70 backdrop-blur">
            <Sparkles className="h-4 w-4 text-accent" />
          </div>
          <div className="font-mono uppercase leading-tight">
            <p className="text-[11px] font-semibold text-foreground">AI Portrait Studio</p>
            <p className="text-[9px] text-muted-foreground">Field Generator</p>
          </div>
        </div>
        <Button
          variant="outline"
          size="sm"
          className="h-8 rounded-full border-border bg-card/60 px-3 font-mono text-[10px] uppercase backdrop-blur"
          onClick={() => setTheme(activeTheme === 'dark' ? 'light' : 'dark')}
        >
          {activeTheme === 'dark' ? <Moon className="h-3.5 w-3.5" /> : <Sun className="h-3.5 w-3.5" />}
          {activeTheme}
        </Button>
      </header>

      <main className="relative z-10 mx-auto flex min-h-screen w-full max-w-7xl flex-col px-4 pb-20 pt-28 md:px-8">
        <section className="mx-auto mb-10 max-w-3xl text-center">
          <Badge variant="accent" className="mb-5 gap-2 rounded-full bg-background/40 px-3 py-1 font-mono">
            <span className="h-1.5 w-1.5 rounded-full bg-accent" />
            THREE.JS / 2,728 POINTS / STUDIO FIELD
          </Badge>
          <h1 className="font-mono text-4xl font-semibold tracking-normal text-balance md:text-6xl">
            AI Portrait Studio
          </h1>
          <p className="mx-auto mt-4 max-w-xl text-sm leading-6 text-muted-foreground md:text-base">
            A professional image editor for turning one portrait into five cinematic, naturally lit studio looks while keeping the face stable.
          </p>
        </section>

        <section className="grid gap-4 lg:grid-cols-[380px_1fr]">
          <Card className="self-start">
            <CardHeader className="border-b border-border">
              <div className="flex items-center justify-between">
                <CardTitle className="font-mono uppercase">Control Surface</CardTitle>
                <Badge variant="muted">{status}</Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-5 pt-4">
              <label
                className="group flex min-h-40 cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed border-border bg-background/55 p-5 text-center transition hover:border-accent/60 hover:bg-accent/5"
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault();
                  handleFile(event.dataTransfer.files[0] ?? null);
                }}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="sr-only"
                  onChange={(event) => handleFile(event.target.files?.[0] ?? null)}
                />
                <Upload className="mb-3 h-6 w-6 text-accent" />
                <span className="text-sm font-semibold">Upload portrait</span>
                <span className="mt-1 max-w-56 overflow-hidden text-ellipsis whitespace-nowrap text-xs text-muted-foreground">
                  {inputState}
                </span>
              </label>

              <div className="grid gap-2">
                <p className="font-mono text-[10px] uppercase text-muted-foreground">Generation Mode</p>
                <div className="grid gap-2">
                  {modes.map((item) => {
                    const Icon = item.icon;
                    const active = mode === item.value;
                    return (
                      <button
                        key={item.value}
                        type="button"
                        className={cn(
                          'flex items-center justify-between rounded-md border border-border bg-background/45 p-3 text-left transition hover:border-accent/60',
                          active && 'border-accent/60 bg-accent/10',
                        )}
                        onClick={() => setMode(item.value)}
                      >
                        <span className="flex items-center gap-3">
                          <Icon className={cn('h-4 w-4 text-muted-foreground', active && 'text-accent')} />
                          <span>
                            <span className="block text-sm font-semibold">{item.label}</span>
                            <span className="block text-xs text-muted-foreground">{item.detail}</span>
                          </span>
                        </span>
                        <span className={cn('h-2 w-2 rounded-full bg-muted', active && 'bg-accent')} />
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="grid grid-cols-[1fr_112px] gap-3">
                <button
                  type="button"
                  className={cn(
                    'flex min-h-12 items-center gap-3 rounded-md border border-border bg-background/45 px-3 text-left text-sm font-semibold',
                    preserveFaces && 'border-accent/50 bg-accent/10',
                  )}
                  onClick={() => setPreserveFaces((current) => !current)}
                >
                  <Lock className={cn('h-4 w-4 text-muted-foreground', preserveFaces && 'text-accent')} />
                  Face lock
                </button>
                <label className="grid gap-1">
                  <span className="font-mono text-[10px] uppercase text-muted-foreground">Seed</span>
                  <input
                    className="h-9 rounded-md border border-input bg-background/50 px-2 text-sm outline-none ring-offset-background focus:ring-1 focus:ring-ring"
                    type="number"
                    value={seed}
                    min={1}
                    onChange={(event) => setSeed(Number(event.target.value))}
                  />
                </label>
              </div>

              <Button className="w-full" onClick={handleGenerate} disabled={isGenerating}>
                <Sparkles className="h-4 w-4" />
                {isGenerating ? 'Creating studio set' : 'Create studio set'}
              </Button>

              <div className="grid gap-2 rounded-lg border border-border bg-background/45 p-3 font-mono text-[10px] uppercase text-muted-foreground">
                <div className="flex justify-between gap-3">
                  <span>GPU</span>
                  <span className="text-right text-foreground">
                    {health?.cuda_available ? health.gpu_name || health.device : 'Unavailable'}
                  </span>
                </div>
                <div className="flex justify-between gap-3">
                  <span>API</span>
                  <span className={cn('text-right', health?.api_key_configured ? 'text-accent' : 'text-muted-foreground')}>
                    {health?.api_key_configured ? `${health.api_image_model} ready` : 'Key missing'}
                  </span>
                </div>
                <div className="flex justify-between gap-3">
                  <span>Review</span>
                  <span className="text-right text-foreground">
                    {selectedCount} / {variations.length} selected
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid gap-4">
            <Card>
              <CardContent className="grid gap-4 p-4 md:grid-cols-[220px_1fr_auto] md:items-center">
                <div className="overflow-hidden rounded-lg border border-border bg-background">
                  <img src={inputPreview} alt="Input portrait" className="aspect-square w-full object-cover" />
                </div>
                <div>
                  <Badge variant="muted" className="mb-3 font-mono">
                    Studio Notes
                  </Badge>
                  <p className="max-w-2xl text-sm leading-6 text-muted-foreground">{summary}</p>
                </div>
                <Button variant="outline" className="md:self-end" onClick={handleSave} disabled={!canSave || isSaving}>
                  <Save className="h-4 w-4" />
                  {isSaving ? 'Saving' : 'Save selection'}
                </Button>
              </CardContent>
            </Card>

            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
              {variations.map((item) => {
                const selected = decisions[item.id]?.decision || '';
                return (
                  <Card key={item.id} className="overflow-hidden">
                    <div className="relative">
                      <img src={item.image_url} alt={item.label} className="aspect-square w-full object-cover" />
                      <Badge variant="muted" className="absolute left-2 top-2 bg-background/70 font-mono backdrop-blur">
                        #{item.id}
                      </Badge>
                    </div>
                    <CardContent className="space-y-3 p-3">
                      <div>
                        <div className="flex items-center justify-between gap-2">
                          <h2 className="truncate text-sm font-semibold">{item.label}</h2>
                          <ImageIcon className="h-3.5 w-3.5 text-muted-foreground" />
                        </div>
                        <p className="mt-1 font-mono text-[10px] uppercase text-muted-foreground">
                          {item.provider ?? 'local'} / strength {item.strength} / seed {item.seed}
                        </p>
                      </div>
                      <p className="font-mono text-[10px] uppercase text-muted-foreground">
                        {item.face_preservation ? `Face lock ${item.detected_faces ?? 0}` : 'Face lock off'}
                      </p>
                      <div className="grid grid-cols-2 gap-2">
                        <Button
                          type="button"
                          size="sm"
                          variant={selected === 'accepted' ? 'default' : 'outline'}
                          onClick={() => setDecision(item.id, 'accepted')}
                          disabled={!sessionId}
                        >
                          <Check className="h-3.5 w-3.5" />
                          Keep
                        </Button>
                        <Button
                          type="button"
                          size="sm"
                          variant={selected === 'rejected' ? 'default' : 'outline'}
                          onClick={() => setDecision(item.id, 'rejected')}
                          disabled={!sessionId}
                        >
                          <X className="h-3.5 w-3.5" />
                          Pass
                        </Button>
                      </div>
                      <input
                        className="h-9 w-full rounded-md border border-input bg-background/50 px-2 text-xs outline-none ring-offset-background focus:ring-1 focus:ring-ring disabled:opacity-50"
                        placeholder="Selection note"
                        value={decisions[item.id]?.reason || ''}
                        disabled={!sessionId}
                        onChange={(event) => setReason(item.id, event.target.value)}
                      />
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </div>
        </section>
      </main>

      <footer className="fixed inset-x-0 bottom-0 z-20 hidden justify-between px-5 py-4 font-mono text-[9px] uppercase text-muted-foreground md:flex">
        <span>Matrix {44} x {62}</span>
        <span>Separation 150 / Palette {activeTheme === 'dark' ? 'light dots' : 'dark dots'}</span>
        <span>Components / Studio / Dotted Surface</span>
      </footer>
    </div>
  );
}
