// Push-optimal Sokoban solver, written from scratch for this bounty.
//
// Strategy: the generator builds every level as a 1-wide hallway with
// independent rooms hanging off it (a box pushed out of a room through its
// door can never come back, so rooms are independent and a level's optimum
// is the sum of its rooms' optima). So:
//   1. split the level into rooms: connected floor components once the
//      hallway row (the row the player starts on) is removed;
//   2. solve each room push-optimally with A* over push states
//      (state = sorted box cells + canonical player region), Hungarian
//      assignment over all-pairs shortest-path distances as the admissible
//      heuristic, corner / 2x2 / frozen-box deadlock pruning, and a
//      transposition table;
//   3. stitch the rooms together with shortest player walks on the full
//      level (walks cost no pushes).
// Everything is budgeted by work (node expansions), never by the clock,
// so two runs produce byte-identical output.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions in fixed order: up, down, left, right.
const DR = [-1, 1, 0, 0];
const DC = [0, 0, -1, 1];
const CH = ['u', 'd', 'l', 'r'];

// ---------------------------------------------------------------- parsing

type Level = {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall, per cell r*w+c
  goals: Set<number>;
  boxes: number[];
  player: number;
};

function parseLevel(text: string): Level {
  const lines = text.split('\n').filter((l) => l.length > 0);
  const h = lines.length;
  const w = Math.max(...lines.map((l) => l.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = new Set<number>();
  const boxes: number[] = [];
  let player = -1;
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < lines[r].length; c++) {
      const ch = lines[r][c];
      const id = r * w + c;
      if (ch === '#') continue;
      wall[id] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.add(id);
      if (ch === '$' || ch === '*') boxes.push(id);
      if (ch === '@' || ch === '+') player = id;
    }
  }
  return { w, h, wall, goals, boxes, player };
}

// ------------------------------------------------------- shortest walk (BFS)

function walkPath(
  w: number,
  wall: Uint8Array,
  blocked: Uint8Array, // 1 = cell occupied by a box (player may not step there)
  from: number,
  to: number,
): string | null {
  if (from === to) return '';
  const n = wall.length;
  const prev = new Int32Array(n).fill(-2);
  const prevDir = new Int8Array(n).fill(-1);
  const q = new Int32Array(n);
  let qh = 0;
  let qt = 0;
  prev[from] = -1;
  q[qt++] = from;
  while (qh < qt) {
    const cur = q[qh++];
    const r = Math.floor(cur / w);
    const c = cur % w;
    for (let d = 0; d < 4; d++) {
      const nr = r + DR[d];
      const nc = c + DC[d];
      if (nr < 0 || nc < 0) continue;
      const nid = nr * w + nc;
      if (nid >= n || wall[nid] || blocked[nid] || prev[nid] !== -2) continue;
      prev[nid] = cur;
      prevDir[nid] = d;
      if (nid === to) {
        let out = '';
        let x = to;
        while (x !== from) {
          out = CH[prevDir[x]] + out;
          x = prev[x];
        }
        return out;
      }
      q[qt++] = nid;
    }
  }
  return null;
}

// ------------------------------------------------------------ room solver

type Push = { from: number; to: number; dir: number }; // cell ids in the level

// Solve one room push-optimally. floorCells are level cell ids (component
// cells plus the hallway entry cell). entry is the level cell id where the
// player starts. Returns the push list, or null if the work budget runs out.
function solveRoom(
  level: Level,
  floorCells: number[],
  entry: number,
  boxes: number[],
  goals: Set<number>,
  budget: { left: number },
): Push[] | null {
  const n = floorCells.length;
  const idx = new Map<number, number>();
  floorCells.forEach((cell, i) => idx.set(cell, i));
  const isGoal = new Uint8Array(n);
  const startBoxes: number[] = [];
  for (const b of boxes) {
    const i = idx.get(b);
    if (i === undefined) return null;
    startBoxes.push(i);
  }
  for (const g of goals) {
    const i = idx.get(g);
    if (i !== undefined) isGoal[i] = 1;
  }
  const entryIdx = idx.get(entry);
  if (entryIdx === undefined) return null;
  startBoxes.sort((a, b) => a - b);

  // Neighbour table nb[i*4+d] = neighbour index or -1.
  const nb = new Int16Array(n * 4).fill(-1);
  for (let i = 0; i < n; i++) {
    const cell = floorCells[i];
    const r = Math.floor(cell / level.w);
    const c = cell % level.w;
    for (let d = 0; d < 4; d++) {
      const nr = r + DR[d];
      const nc = c + DC[d];
      if (nr < 0 || nc < 0) continue;
      const nid = nr * level.w + nc;
      const j = idx.get(nid);
      if (j !== undefined) nb[i * 4 + d] = j;
    }
  }

  // All-pairs shortest path distances over the room floor (walls only).
  const INF = 1e9;
  const dist = new Int32Array(n * n);
  {
    const q = new Int32Array(n);
    for (let s = 0; s < n; s++) {
      for (let j = 0; j < n; j++) dist[s * n + j] = INF;
      dist[s * n + s] = 0;
      let qh = 0;
      let qt = 0;
      q[qt++] = s;
      while (qh < qt) {
        const cur = q[qh++];
        const cd = dist[s * n + cur];
        for (let d = 0; d < 4; d++) {
          const m = nb[cur * 4 + d];
          if (m >= 0 && dist[s * n + m] === INF) {
            dist[s * n + m] = cd + 1;
            q[qt++] = m;
          }
        }
      }
    }
  }

  // Static dead cells: not a goal, and either a corner (two perpendicular
  // wall neighbours) or unable to reach any goal at all.
  const dead = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (isGoal[i]) continue;
    let reachGoal = false;
    for (let g = 0; g < n; g++) if (isGoal[g] && dist[i * n + g] < INF) { reachGoal = true; break; }
    if (!reachGoal) { dead[i] = 1; continue; }
    const up = nb[i * 4 + 0] < 0;
    const down = nb[i * 4 + 1] < 0;
    const left = nb[i * 4 + 2] < 0;
    const right = nb[i * 4 + 3] < 0;
    if ((up || down) && (left || right)) dead[i] = 1;
  }

  const k = startBoxes.length;
  const goalList: number[] = [];
  for (let i = 0; i < n; i++) if (isGoal[i]) goalList.push(i);

  // Hungarian heuristic by brute-force assignment (k <= 4, so <= 24 perms).
  const hCache = new Map<number, number>();
  function packBoxes(b: number[]): number {
    let p = 0;
    for (let i = 0; i < k; i++) p |= b[i] << (6 * i);
    for (let i = k; i < 4; i++) p |= 63 << (6 * i);
    return p;
  }
  function heuristic(b: number[]): number {
    const p = packBoxes(b);
    const cached = hCache.get(p);
    if (cached !== undefined) return cached;
    let best = INF;
    // Iterate all injective assignments boxes -> goals via permutations of
    // goal indices (goalList length == k in these rooms).
    const m = goalList.length;
    const perm = new Array<number>(k);
    const used = new Uint8Array(m);
    const rec = (bi: number, sum: number): void => {
      if (sum >= best) return;
      if (bi === k) { if (sum < best) best = sum; return; }
      for (let gi = 0; gi < m; gi++) {
        if (used[gi]) continue;
        const d = dist[b[bi] * n + goalList[gi]];
        if (d >= INF) continue;
        used[gi] = 1;
        perm[bi] = gi;
        rec(bi + 1, sum + d);
        used[gi] = 0;
      }
    };
    rec(0, 0);
    hCache.set(p, best);
    return best;
  }

  // Reachable region from a player cell, given box occupancy. Returns the
  // stamp array generation and the smallest reachable index (canonical
  // player position). Uses a shared stamp array to avoid allocations.
  const stamp = new Uint32Array(n);
  let gen = 0;
  const reachQueue = new Int32Array(n);
  function reachable(player: number, occ: Uint8Array): { stampGen: number; min: number } {
    gen++;
    let qh = 0;
    let qt = 0;
    stamp[player] = gen;
    reachQueue[qt++] = player;
    let min = player;
    while (qh < qt) {
      const cur = reachQueue[qh++];
      for (let d = 0; d < 4; d++) {
        const m = nb[cur * 4 + d];
        if (m >= 0 && !occ[m] && stamp[m] !== gen) {
          stamp[m] = gen;
          if (m < min) min = m;
          reachQueue[qt++] = m;
        }
      }
    }
    return { stampGen: gen, min };
  }

  // A* nodes. Node arrays are parallel; heap holds node indices.
  const nodePack: number[] = [];
  const nodePlayer: number[] = [];
  const nodeParent: number[] = [];
  const nodePushFrom: number[] = [];
  const nodePushTo: number[] = [];
  const nodePushDir: number[] = [];
  const nodeG: number[] = [];
  const nodeF: number[] = [];
  const best = new Map<number, number>(); // state key -> best g seen

  function unpackBoxes(p: number, out: number[]): void {
    for (let i = 0; i < k; i++) out[i] = (p >> (6 * i)) & 63;
  }
  function stateKey(pack: number, player: number): number {
    return pack | (player << 24);
  }

  const heap: number[] = [];
  function heapLess(a: number, b: number): boolean {
    if (nodeF[a] !== nodeF[b]) return nodeF[a] < nodeF[b];
    return nodeG[a] > nodeG[b]; // deeper first on ties
  }
  function heapPush(v: number): void {
    heap.push(v);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i], heap[p])) break;
      const t = heap[i]; heap[i] = heap[p]; heap[p] = t;
      i = p;
    }
  }
  function heapPop(): number {
    const top = heap[0];
    const last = heap.pop() as number;
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = i * 2 + 1;
        const r = l + 1;
        let m = i;
        if (l < heap.length && heapLess(heap[l], heap[m])) m = l;
        if (r < heap.length && heapLess(heap[r], heap[m])) m = r;
        if (m === i) break;
        const t = heap[i]; heap[i] = heap[m]; heap[m] = t;
        i = m;
      }
    }
    return top;
  }

  function addNode(
    pack: number, player: number, parent: number,
    pf: number, pt: number, pd: number, g: number, f: number,
  ): number {
    const id = nodePack.length;
    nodePack.push(pack);
    nodePlayer.push(player);
    nodeParent.push(parent);
    nodePushFrom.push(pf);
    nodePushTo.push(pt);
    nodePushDir.push(pd);
    nodeG.push(g);
    nodeF.push(f);
    return id;
  }

  const startOcc = new Uint8Array(n);
  for (const b of startBoxes) startOcc[b] = 1;
  const startReach = reachable(entryIdx, startOcc);
  const startPack = packBoxes(startBoxes);
  const startH = heuristic(startBoxes);
  if (startH >= INF) return null;
  const root = addNode(startPack, startReach.min, -1, -1, -1, -1, 0, startH);
  best.set(stateKey(startPack, startReach.min), 0);
  heapPush(root);

  const curBoxes = new Array<number>(k);
  const occ = new Uint8Array(n);
  const reachMask = new Uint8Array(n);
  let expansions = 0;
  let goalNode = -1;

  while (heap.length > 0) {
    if (budget.left <= 0) return null;
    const node = heapPop();
    const g = nodeG[node];
    if (g !== best.get(stateKey(nodePack[node], nodePlayer[node]))) continue;
    expansions++;
    budget.left--;
    unpackBoxes(nodePack[node], curBoxes);
    // Goal test: every box on a goal.
    let solved = true;
    for (let i = 0; i < k; i++) if (!isGoal[curBoxes[i]]) { solved = false; break; }
    if (solved) { goalNode = node; break; }
    for (let i = 0; i < n; i++) occ[i] = 0;
    for (let i = 0; i < k; i++) occ[curBoxes[i]] = 1;
    const reach = reachable(nodePlayer[node], occ);
    // Snapshot the region: later reachable() calls inside this expansion
    // reuse the stamp array and would overwrite it.
    for (let i = 0; i < n; i++) reachMask[i] = stamp[i] === reach.stampGen ? 1 : 0;
    for (let bi = 0; bi < k; bi++) {
      const b = curBoxes[bi];
      for (let d = 0; d < 4; d++) {
        const behind = nb[b * 4 + (d ^ 1)];
        const target = nb[b * 4 + d];
        if (behind < 0 || target < 0) continue;
        if (!reachMask[behind]) continue; // player cannot get behind
        if (occ[target]) continue;
        // Evaluate deadlocks in the NEW configuration (box moved to target).
        occ[b] = 0;
        occ[target] = 1;
        if (dead[target]) { occ[target] = 0; occ[b] = 1; continue; }
        // 2x2 freeze: any 2x2 block containing the target cell that is
        // entirely wall-or-box with no goal in it.
        if (frozen2x2(target)) { occ[target] = 0; occ[b] = 1; continue; }
        // Frozen against a wall: a wall on one side and both cells along
        // the wall blocked (wall or box), box not on a goal.
        if (frozenWall(target, occ)) { occ[target] = 0; occ[b] = 1; continue; }
        const nb2 = curBoxes.slice();
        nb2[bi] = target;
        nb2.sort((a, b2) => a - b2);
        // New player position: the box's old cell, canonicalised.
        const nr = reachable(b, occ);
        occ[target] = 0;
        occ[b] = 1;
        const npack = packBoxes(nb2);
        const key = stateKey(npack, nr.min);
        const ng = g + 1;
        const prevBest = best.get(key);
        if (prevBest !== undefined && prevBest <= ng) continue;
        const h = heuristic(nb2);
        if (h >= INF) continue;
        best.set(key, ng);
        const id = addNode(npack, nr.min, node, b, target, d, ng, ng + h);
        heapPush(id);
      }
    }
  }
  if (goalNode < 0) return null;

  function frozen2x2(t: number): boolean {
    // t's level cell; find its room-grid coords via floorCells.
    const cell = floorCells[t];
    const r = Math.floor(cell / level.w);
    const c = cell % level.w;
    for (let dr = -1; dr <= 0; dr++) {
      for (let dc = -1; dc <= 0; dc++) {
        let allBlocked = true;
        let hasGoal = false;
        for (let a = 0; a < 2; a++) {
          for (let b2 = 0; b2 < 2; b2++) {
            const nr = r + dr + a;
            const nc = c + dc + b2;
            const nid = nr * level.w + nc;
            const j = idx.get(nid);
            if (j === undefined) continue; // wall
            if (isGoal[j]) hasGoal = true;
            if (!occ[j]) allBlocked = false;
          }
        }
        if (allBlocked && !hasGoal) return true;
      }
    }
    return false;
  }
  function frozenWall(t: number, o: Uint8Array): boolean {
    if (isGoal[t]) return false;
    // For each side that is a wall, check the two cells along that wall.
    for (let d = 0; d < 4; d++) {
      if (nb[t * 4 + d] >= 0) continue; // not a wall on this side
      const perp = d < 2 ? [2, 3] : [0, 1];
      let blocked = 0;
      for (const pd of perp) {
        const m = nb[t * 4 + pd];
        if (m < 0 || o[m]) blocked++;
      }
      if (blocked === 2) return true;
    }
    return false;
  }

  // Reconstruct the push list (level cell ids).
  const pushes: Push[] = [];
  let cur = goalNode;
  while (cur > 0) {
    pushes.push({
      from: floorCells[nodePushFrom[cur]],
      to: floorCells[nodePushTo[cur]],
      dir: nodePushDir[cur],
    });
    cur = nodeParent[cur];
  }
  pushes.reverse();
  void expansions;
  return pushes;
}

// ------------------------------------------------------------ whole level

function solveLevel(text: string): string {
  const level = parseLevel(text);
  const { w, wall } = level;
  const hallwayRow = Math.floor(level.player / w);
  const n = w * level.h;

  // Connected floor components excluding the hallway row.
  const comp = new Int32Array(n).fill(-1);
  const rooms: { cells: number[]; boxes: number[]; compId: number }[] = [];
  let ncomp = 0;
  for (let id = 0; id < n; id++) {
    if (wall[id] || Math.floor(id / w) === hallwayRow || comp[id] >= 0) continue;
    const cells: number[] = [];
    const q = [id];
    comp[id] = ncomp;
    while (q.length > 0) {
      const cur = q.pop() as number;
      cells.push(cur);
      const r = Math.floor(cur / w);
      const c = cur % w;
      for (let d = 0; d < 4; d++) {
        const nr = r + DR[d];
        const nc = c + DC[d];
        if (nr < 0 || nc < 0) continue;
        const nid = nr * w + nc;
        if (nid >= n || wall[nid] || Math.floor(nid / w) === hallwayRow || comp[nid] >= 0) continue;
        comp[nid] = ncomp;
        q.push(nid);
      }
    }
    const boxes = level.boxes.filter((b) => comp[b] === ncomp);
    if (boxes.length > 0) rooms.push({ cells, boxes, compId: ncomp });
    ncomp++;
  }
  if (rooms.length === 0) return '';

  // Deterministic room order.
  rooms.sort((a, b) => Math.min(...a.cells) - Math.min(...b.cells));

  const budget = { left: 4000000 }; // total push-expansions for the level
  const curBoxes = level.boxes.slice();
  let playerPos = level.player;
  let moves = '';

  const blocked = new Uint8Array(n);
  const markBlocked = (): void => {
    blocked.fill(0);
    for (const b of curBoxes) blocked[b] = 1;
  };

  for (const room of rooms) {
    const roomGoals = new Set<number>();
    for (const g of level.goals) if (comp[g] === room.compId) roomGoals.add(g);
    // Entry candidates: hallway-row floor cells adjacent to the room.
    const entrySet = new Set<number>();
    for (const cell of room.cells) {
      const r = Math.floor(cell / w);
      const c = cell % w;
      for (let d = 0; d < 4; d++) {
        const nr = r + DR[d];
        const nc = c + DC[d];
        if (nr < 0 || nc < 0) continue;
        const nid = nr * w + nc;
        if (nid < n && !wall[nid] && Math.floor(nid / w) === hallwayRow) entrySet.add(nid);
      }
    }
    const entries = [...entrySet].sort((a, b) => a - b);
    let pushes: Push[] | null = null;
    let entry = -1;
    for (const e of entries) {
      const floor = room.cells.concat([e]);
      const p = solveRoom(level, floor, e, room.boxes, roomGoals, budget);
      if (p !== null) { pushes = p; entry = e; break; }
    }
    if (pushes === null || entry < 0) return '';
    // Walk from the current player position to the room entry.
    markBlocked();
    const toEntry = walkPath(w, wall, blocked, playerPos, entry);
    if (toEntry === null) return '';
    moves += toEntry;
    playerPos = entry;
    // Execute the room's pushes, walking between them.
    for (const push of pushes) {
      markBlocked();
      const behind = push.from - (DR[push.dir] * w + DC[push.dir]);
      const walk = walkPath(w, wall, blocked, playerPos, behind);
      if (walk === null) return '';
      moves += walk;
      moves += CH[push.dir];
      playerPos = push.from;
      const bi = curBoxes.indexOf(push.from);
      if (bi < 0) return '';
      curBoxes[bi] = push.to;
    }
  }

  // Safety net: replay the whole move string; only keep it if it solves.
  const simBoxes = new Set(level.boxes);
  let px = level.player;
  for (const ch of moves) {
    const d = CH.indexOf(ch);
    const r = Math.floor(px / w);
    const c = px % w;
    const nr = r + DR[d];
    const nc = c + DC[d];
    const nid = nr * w + nc;
    if (nid < 0 || nid >= n || wall[nid]) return '';
    if (simBoxes.has(nid)) {
      const beyond = (nr + DR[d]) * w + (nc + DC[d]);
      if (beyond < 0 || beyond >= n || wall[beyond] || simBoxes.has(beyond)) return '';
      simBoxes.delete(nid);
      simBoxes.add(beyond);
    }
    px = nid;
  }
  for (const b of simBoxes) if (!level.goals.has(b)) return '';
  return moves;
}

for (const id of readdirSync(IN).sort()) {
  const moves = solveLevel(readFileSync(join(IN, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUT, id), { recursive: true });
  writeFileSync(join(OUT, id, 'solution.txt'), `${moves}\n`);
}
