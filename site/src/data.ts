// The files pipeline/export.py writes, and typed loaders for them.

export type Status = "pool" | "held back" | "searched" | "rejected" | "flagged" | "awaiting agent" | "candidate" | "known";

export interface Sky {
  tic: number[]; ra: number[]; dec: number[]; dist: number[]; tmag: number[]; teff: number[]; status: number[];
  codes: Record<Status, number>;
}

export interface Summary {
  pool: number;
  funnel: { step: string; stars: number }[];
  held_back: number;
  by_status: Record<Status, number>;
  costs_usd: Record<"vetting" | "investigation" | "case_closed" | "observing" | "audit", number>;
  invented?: { dossiers: number; with_invented_numbers: number };
  dossiers: number;
  investigated: number;
  rejected: number; flagged: number; candidate: number;
  [k: string]: unknown;
}

export interface Dip {
  tic: number; batch: string; status: Status; period_d: number; depth_ppm: number; snr: number; rp_rearth: number | null;
  tmag: number | null; teff: number | null; dist_pc: number | null; ra: number; dec: number; line: string; reasons: string[];
  dossier: boolean; tools: number;
}

export interface Reason { check: string; value: string; why: string }
export interface Dossier {
  verdict: string; reasons: Reason[]; open_questions?: string[]; public_summary?: string;
  model: string; prompt?: string; cost_usd: number; tools_called?: string[]; invented?: number[];
}

export interface Ground { site: string; mid_utc: string; from_utc: string; to_utc: string; timing_err_min: number; alt_mid: number }
export interface TessSector {
  sector: number; camera: number; ccd: number; start: string; end: string; public_from: string; transits: number;
  timing_err_min: number; lost: boolean;
}

// The checks are open-ended dicts from pipeline/vet.py; the page reads them defensively.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Checks = Record<string, any>;

export interface StarPage {
  tic: number; batch: string; status: Status; reasons: string[];
  gate1: { dip_snr: number; flipped_snr: number; transits: number };
  case_closed: string | null;
  ephem: { period: number; t0: number };
  checks: Checks;
  dossier: Dossier | null;
  series: { t: number[]; y_ppm: number[]; sector: number[] };
  fold: { x_h: number[]; y_ppm: number[]; err_ppm: number[] };
  evidence?: Record<string, Checks>;
  novelty?: { new: boolean; complete: boolean; matches: Checks[]; notes: string[]; errors: string[] };
  when?: {
    ephemeris: { period: number; period_err: number; t0_btjd: number; t0_err_min: number; t14_h: number };
    timing_err_now_min: number; testable: boolean; next_tess: TessSector | null; tess_sectors: TessSector[];
    ground_nights: number; ground: Ground[]; ground_snr_per_transit: Record<string, number>;
  };
  observing?: { headline: string; next_chances: string[]; telescope: string; refuted_if: string; confirmed_by: string; model: string; cost_usd: number };
  audit?: Dossier;
}

// planets.json: kind 0 confirmed, 1 TESS candidate still waiting, 2 Tessa's candidate; method and facility index the lists
export interface Planets {
  methods: string[]; facilities: string[];
  counts: { confirmed: number; waiting: number; tessa: number; unplaced: { confirmed: number; waiting: number };
    by_method: Record<string, { planets: number; first: number }> };
  kind: number[]; name: string[]; host: string[]; ra: number[]; dec: number[]; dist: number[]; method: number[];
  year: number[]; facility: number[]; period: (number | null)[]; radius: (number | null)[]; mass: (number | null)[];
  teq: (number | null)[]; teff: (number | null)[];
}

const cache = new Map<string, Promise<unknown>>();

function load<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    cache.set(path, fetch(path).then((r) => {
      if (!r.ok) throw new Error(`${path}: ${r.status}`);
      return r.json();
    }).catch((e) => {
      cache.delete(path);
      throw e;
    }));
  }
  return cache.get(path) as Promise<T>;
}

export const getSky = () => load<Sky>("/data/sky.json");
export const getPlanets = () => load<Planets>("/data/planets.json");
export const getSummary = () => load<Summary>("/data/summary.json");
export const getDips = () => load<Dip[]>("/data/dips.json");
export const getStar = (tic: number) => load<StarPage>(`/data/stars/${tic}.json`);
