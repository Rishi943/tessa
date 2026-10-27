// The tour: seven stops through what Tessa did during the hackathon. Each stop sets the universe behind its card
// (where the camera goes, how far the discovery clock runs, which planets are lit) and names the card it shows.
import type { CardKind } from "./tourcards";

export type Group = "transit" | "rv" | "lens" | "other" | "waiting" | "tessa";

export interface Stop {
  card: CardKind;
  at?: "ours";      // fly to Tessa's candidate; without it the view centres on the Sun
  dist?: number;    // how far back the camera sits, in parsecs (by default it fits the Sun and the candidate)
  year: number;     // the discovery clock: later planets stay hidden (waiting candidates count as 2027, Tessa's 2028)
  focus?: Group;    // light these up and dim the rest
  patch?: boolean;  // draw the edge of Tessa's patch
  dim?: boolean;    // darken the universe behind a card that carries its own picture
}

export const STOPS: Stop[] = [
  { card: "hook", at: "ours", year: 3000, focus: "tessa" },
  { card: "dip", year: 2026.9, focus: "transit", dim: true },
  { card: "fold", dist: 230, year: 3000, patch: true, dim: true },
  { card: "odds", dist: 230, year: 3000, patch: true, dim: true },
  { card: "casefile", year: 2027.9, focus: "waiting" },
  { card: "predict", at: "ours", dist: 6, year: 3000, dim: true },
  { card: "end", year: 3000 },
];
