// mimo.js: "MiMo on the 170HX" — a ~34 s papery explainer, one shot per lyric line (see ../../STORYBOARD.md).
//   A wake · B three cards · C slice the brain · D pipeline relay · E guess-ahead helper
//   F decode 117 · G prefill 4,000 · H 32 streams 561 · I rhyme ending
// Shot start times live in START (snapped to the chosen take's word timestamps); every shot times its events
// as fractions of its own length, so re-snapping never breaks a shot.
(() => {
  // snapped to take mimo_synthpop_s11 (audio trimmed from 6.6 s): each shot starts on its lyric line
  const START = { A: 0, B: 1.92, C: 6.36, D: 10.24, E: 13.6, F: 18.0, G: 21.96, H: 26.22, I: 29.8 };

  // ---------- palette ----------
  const C = {
    wall: '#3E4A5C', wallLt: '#566579', shelf: '#7A5C44', shelfDk: '#5A4232', bench: '#8A6A4E',
    pcb: '#7FA38A', pcbDk: '#58786A', shroud: '#46525E', gold: '#FFD27A', goldDk: '#E0A94A',
    brain: '#E6A2B4', brainDk: '#B9748B', paper: PAL.cream, ink: PAL.ink,
  };
  const CARDS = [[220, 505], [540, 505], [860, 505]];   // card centres on the shelf
  const CW = 270, CH = 108, FLOOR = 930;

  // ---------- set pieces ----------
  function room(t, warm = 0) {
    boilSeed('wall');
    paint(rectPts(-300, -300, W + 600, H + 600), { wash: mixCol(C.wall, '#6A5A48', warm * .35), ink: null });
    paint(ellPts(W * .5, H * .38, W * .75, H * .45, 30, 12), { fill: mixCol(C.wallLt, C.goldDk, warm * .5), fillOp: 70 + 60 * warm, bleed: .3, tex: .6, ink: null });
    boilSeed('shelf');
    paint(rectPts(40, 560, W - 80, 26, 1.5), { wash: C.shelf, ink: C.ink, sw: .9 });
    paint(rectPts(70, 586, 22, 60, 1), { wash: C.shelfDk, ink: C.ink, sw: .7 });
    paint(rectPts(W - 92, 586, 22, 60, 1), { wash: C.shelfDk, ink: C.ink, sw: .7 });
    boilSeed('bench');
    paint(rectPts(-100, FLOOR, W + 200, H - FLOOR + 200, 2), { wash: C.bench, fill: mixCol(C.bench, C.shelfDk, .5), fillOp: 90, bleed: .05, tex: .7, ink: C.ink, sw: 1 });
  }
  // one fan: a ring plus blades turning at `ang`; fast fans smear into a disc
  function fan(x, y, r, ang, speed) {
    paint(ellPts(x, y, r, r, 22, .6), { wash: mixCol(C.shroud, '#2E3740', .4), ink: C.ink, sw: .8 });
    if (speed > 6) paint(ellPts(x, y, r * .86, r * .86, 20, .4), { fill: '#A9B4BE', fillOp: 90, bleed: .1, tex: .5, ink: null });
    const n = 7;
    for (let b = 0; b < n; b++) {
      const a = ang + b * TAU / n;
      inkLine([[x + Math.cos(a) * r * .22, y + Math.sin(a) * r * .22], [x + Math.cos(a + .5) * r * .8, y + Math.sin(a + .5) * r * .8]], speed > 6 ? .4 : .9, speed > 6 ? '#8C98A3' : C.ink, 'ink', .5);
    }
    paint(ellPts(x, y, r * .2, r * .2, 12, .3), { wash: C.shroud, ink: C.ink, sw: .6 });
  }
  // a mining card: sage PCB with shroud and two fans. on 0..1 = asleep → glowing; spin = fan speed (rad/s)
  function card(i, t, on = 0, spin = 0, o = {}) {
    const [cx, cy] = CARDS[i], x = cx - CW / 2, y = cy - CH / 2;
    if (on > 0) glow(cx, cy, 130 + 60 * on, C.gold, .22 + .55 * on);
    boilSeed('card' + i);
    paint(rrPts(x, y, CW, CH, 10, 1), { wash: mixCol(C.pcbDk, C.pcb, .35 + .65 * Math.min(1, on + .2)), ink: C.ink, sw: 1 });
    paint(rrPts(x + 10, y + 10, CW - 20, CH - 20, 8, .8), { wash: C.shroud, ink: C.ink, sw: .7 });
    paint(rectPts(x + 18, y + CH - 4, CW - 36, 10, .5), { wash: C.goldDk, ink: C.ink, sw: .5 });   // gold fingers
    const ang = spin * t + i * 1.3;
    fan(cx - 62, cy, 38, ang, spin); fan(cx + 62, cy, 38, -ang * .97, spin);
    if (o.chips != null) for (let k = 0; k < 8; k++) {   // memory chips filling up along the top edge
      const f = clamp(o.chips * 8 - k);
      paint(rectPts(x + 20 + k * 29, y - 14, 22, 12, .4), { wash: mixCol(C.shroud, C.gold, f), ink: C.ink, sw: .5 });
    }
    if (on > .05) { boilSeed('edge' + i); inkLine([[x + 6, y + 2], [x + CW - 6, y + 2]], 1.2 * on, C.gold, 'ink', 0); }
  }
  // the brain: a lumpy cloud of expert pebbles. s = scale, slice 0..1 = cut into three pieces spread apart
  function brain(cx, cy, s, t, slice = 0, glowK = .8) {
    glow(cx, cy, 230 * s, '#F5B8C8', .35 * glowK);
    for (let p = 0; p < 3; p++) {
      const ox = (p - 1) * 70 * s * slice * 1.4;
      boilSeed('brainlobe' + p);
      const lx = cx + (p - 1) * 88 * s + ox, ly = cy + (p === 1 ? -22 * s : 10 * s), B = [];
      for (let q = 0; q < 44; q++) {   // a bumpy lobe outline, so the mass reads as a brain, not a disc
        const a = q / 44 * TAU, r = (1 + .07 * Math.sin(a * 11 + p * 2) + .04 * Math.sin(a * 5 + p));
        B.push([lx + Math.cos(a) * 112 * s * r, ly + Math.sin(a) * 98 * s * r]);
      }
      paint(B, { wash: C.brain, fill: C.brainDk, fillOp: 70, bleed: .1, tex: .6, ink: C.ink, sw: clamp(1.1 * s, .5, 1.3), curv: .6 });
      for (let f = 0; f < 6; f++) {   // folds (sulci), so it reads as a brain
        const bx = cx + (p - 1) * 88 * s + ox, by = cy + (p === 1 ? -22 * s : 10 * s), a0 = hash(f + p * 9) * TAU;
        const P = []; for (let q = 0; q < 6; q++) { const a = a0 + q * .45, rr = (35 + 45 * hash(f * 7 + q + p)) * s; P.push([bx + Math.cos(a) * rr, by + Math.sin(a) * rr * .75 + Math.sin(q * 2.1 + f) * 9 * s]); }
        inkLine(P, clamp(.9 * s, .4, 1.1), C.brainDk, 'ink', .8);
      }
      for (let k = 0; k < 16; k++) {   // pebbles = experts
        const a = hash(k + p * 40) * TAU, rr = Math.sqrt(hash(k + p * 40 + 7)) * 80 * s;
        const px = cx + (p - 1) * 88 * s + ox + Math.cos(a) * rr, py = cy + (p === 1 ? -22 * s : 10 * s) + Math.sin(a) * rr * .8;
        const tw = .6 + .4 * Math.sin(t * 3 + k + p);
        paint(ellPts(px, py, 11 * s, 9 * s, 10, .6 * s), { wash: mixCol(C.brainDk, '#FFE3EA', tw), ink: null });
      }
    }
  }
  function bead(x, y, r, k = 1) {
    if (k <= 0) return;
    glow(x, y, r * 3.2, C.gold, .6 * k);
    paint(ellPts(x, y, r * k, r * k, 12, .3), { wash: C.gold, fill: '#FFF1C6', fillOp: 120, ink: C.ink, sw: .5 });
  }
  // a bead's position along the relay: card 0 → 1 → 2, hopping on arcs. p 0..1 over the whole row
  function relayPt(p) {
    const q = clamp(p) * 2, i = Math.min(1, Math.floor(q)), k = q - i;
    const a = [CARDS[i][0], CARDS[i][1] - 70], b = [CARDS[i + 1][0], CARDS[i + 1][1] - 70];
    return arcPt(a, b, 70, ease(k));
  }
  const sparkle = (x, y, r, k) => { if (k > 0 && k < 1) paint(starPts(x, y, r * backOut(k) * (1 - k * .6), .25, 4, k * 2), { wash: PAL.cream, washOp: 255 * (1 - k * k), ink: null }); };
  const label = (txt, x, y, size, k, o = {}) => { if (k > 0) letter(txt, x, y, size, o.color || C.gold, { pop: k * 1.6, stroke: C.ink, rot: o.rot ?? -.04, ...o }); };
  const fanSpin = (t, t0, t1, top) => top * easeIn(seg(t, t0, t1));


  // ---------- Mitu, the Mi Bunny (redrawn, no logo): white bunny, cream/green ushanka, red scarf ----------
  // mitu(x, y, u, o): (x, y) = ground point; ~11u tall. o: face, lookX/lookY, aL/aR (0 = out, + = up), armR(u, sw),
  // dy/sq (in u / squash), walk (bob phase), flip, tint ('gold'), emote/emoteK/emoteAge, boilKey, noShadow
  const M = { white: '#EEF3F6', shade: '#C4D2DB', pink: '#F2A7B0', olive: '#5E6B3A', oliveDk: '#465129', brim: '#F2C14E', red: '#D6453A', redDk: '#A8302A', shirt: '#F07A2E', shirtDk: '#C95E1C', shorts: '#4E5E36' };
  const FACES = {
    sleepy: { eyes: 'closed', mouth: 'smile', emote: 'zzz' },  surprised: { eyes: 'wide', mouth: 'o', emote: '!' },
    happy: { eyes: 'happy', mouth: 'smile' },                  scared: { eyes: 'wide', mouth: 'o', emote: 'sweat' },
    determined: { eyes: 'narrow', mouth: 'flat', emote: 'steam' }, proud: { eyes: 'happy', mouth: 'smile', emote: 'spark' },
    excited: { eyes: 'wide', mouth: 'grin', emote: 'spark' },  starstruck: { eyes: 'star', mouth: 'o', emote: 'stars' },
    laugh: { eyes: 'happy', mouth: 'open' },
  };
  const feelM = (name, t, over = {}) => ({ face: name, dy: -.12 * Math.abs(Math.sin(bpOf(t) * Math.PI)), emote: FACES[name].emote, emoteK: 1, emoteAge: t, ...over });
  function emotionsM(lt, keys, o = {}) {   // acted mood changes: a squash take on each switch, the new emote pops in
    let i = 0; while (i + 1 < keys.length && lt >= keys[i + 1][0]) i++;
    const [k0, name, over] = keys[i], tk = i > 0 ? take(lt, k0, o.take ?? .8) : { sq: 0, dy: 0 };
    const base = feelM(name, lt, over || {});
    return { ...base, sq: tk.sq || 0, dy: base.dy + (tk.dy || 0), emoteK: seg(lt, k0, k0 + .3), emoteAge: lt - k0 };
  }
  function mitu(x, y, u, o = {}) {
    const f = FACES[o.face || 'happy'], key = o.boilKey || 'mitu';
    const bob = o.walk != null ? -.35 * Math.abs(Math.sin(o.walk)) : 0;
    const col = c => o.tint === 'gold' ? mixCol(c, '#FFD27A', .55) : c;
    const sw = clamp(u / 22, .45, 1.1);
    if (!o.noShadow) { boilSeed(key + 'sh'); paint(ellPts(x, y + 2, 3.4 * u, .5 * u, 16, .3), { fill: PAL.ink, fillOp: 50, bleed: .1, ink: null }); }
    push(); translate(x, y + ((o.dy || 0) + bob) * u); scale((o.flip ? -1 : 1) * (1 + (o.sq || 0) * .5), 1 - (o.sq || 0));
    const P = (px, py, rx, ry, rot = 0, n = 20) => ellPts(px * u, py * u, rx * u, ry * u, n, .35 * u / 22, rot);
    boilSeed(key + 'ears');   // two tall upright ears, pink inside
    for (const s of [-1, 1]) {
      paint(P(s * 1.5, -14.2, .85, 2.5, s * .18), { wash: col(M.white), ink: PAL.ink, sw });
      paint(P(s * 1.52, -14.0, .38, 1.7, s * .18), { wash: M.pink, ink: null });
    }
    boilSeed(key + 'legs');
    for (const s of [-1, 1]) paint(P(s * 1.25, -.75, .95, 1.0), { wash: col(M.white), fill: M.shade, fillOp: 60, ink: PAL.ink, sw });
    boilSeed(key + 'body');   // shorts, then the orange tee
    paint(P(0, -2.3, 2.45, 1.4, 0, 20), { wash: M.shorts, ink: PAL.ink, sw });
    paint(P(0, -3.9, 2.35, 1.75, 0, 22), { wash: M.shirt, fill: M.shirtDk, fillOp: 50, bleed: .05, tex: .5, ink: PAL.ink, sw });
    boilSeed(key + 'scarf');   // red neckerchief: band + triangle
    paint(P(0, -5.55, 2.2, .45, 0, 16), { wash: M.red, ink: PAL.ink, sw });
    paint([[-1.1 * u, -5.4 * u], [1.1 * u, -5.4 * u], [.1 * u, -3.9 * u]], { wash: M.red, fill: M.redDk, fillOp: 60, ink: PAL.ink, sw });
    boilSeed(key + 'head');   // big chibi head
    paint(P(0, -8.7, 3.85, 3.35, 0, 28), { wash: col(M.white), fill: M.shade, fillOp: 40, bleed: .05, tex: .4, ink: PAL.ink, sw });
    boilSeed(key + 'hat');   // olive ushanka: crown, yellow brim with a red star, flaps down the sides
    paint(P(0, -11.3, 3.9, 1.95, 0, 22), { wash: M.olive, fill: M.oliveDk, fillOp: 70, ink: PAL.ink, sw });
    paint(rrPts(-3.6 * u, -11.2 * u, 7.2 * u, 1.1 * u, .55 * u, .3), { wash: M.brim, ink: PAL.ink, sw });
    paint(starPts(0, -10.65 * u, .62 * u, .45, 5, -Math.PI / 2), { wash: M.red, ink: PAL.ink, sw: sw * .6 });
    for (const s of [-1, 1]) {
      paint(P(s * 3.7, -8.2, .85, 2.5, -s * .18), { wash: M.olive, fill: M.oliveDk, fillOp: 60, ink: PAL.ink, sw });
      paint(P(s * 3.55, -8.0, .35, 1.8, -s * .18), { wash: M.brim, ink: null });
    }
    for (const [s, a0, cb] of [[-1, o.aL ?? -.4, o.armL], [1, o.aR ?? -.4, o.armR]]) {   // arms: orange sleeve, white hand
      boilSeed(key + 'arm' + s);
      const sx = s * 2.0 * u, sy = -4.6 * u, tx = sx + s * Math.cos(a0) * 1.9 * u, ty = sy - Math.sin(a0) * 1.9 * u;
      paint(ribbon([[sx, sy], [tx, ty]], .95 * u, .75 * u), { wash: col(M.white), ink: PAL.ink, sw });
      paint(ellPts(lerp(sx, tx, .25), lerp(sy, ty, .25), .6 * u, .55 * u, 12, .2), { wash: M.shirt, ink: PAL.ink, sw: sw * .8 });
      paint(ellPts(tx, ty, .55 * u, .55 * u, 12, .2), { wash: col(M.white), ink: PAL.ink, sw });
      if (cb) { push(); translate(tx, ty); rotate(-a0 * s); cb(u, sw); pop(); }
    }
    boilSeed(key + 'blush');
    for (const s of [-1, 1]) paint(P(s * 2.1, -7.3, .6, .32), { fill: M.pink, fillOp: 150, bleed: .2, ink: null });
    boilSeed(key + 'face');
    const lx = (o.lookX || 0) * .3, ly = (o.lookY || 0) * .35;
    for (const s of [-1, 1]) {
      const ex = s * 1.15 * u, ey = -8.5 * u;
      if (f.eyes === 'closed') inkLine([[ex - .6 * u, ey], [ex, ey + .35 * u], [ex + .6 * u, ey]], sw, PAL.ink, 'ink', .7);
      else if (f.eyes === 'happy') inkLine([[ex - .6 * u, ey + .2 * u], [ex, ey - .4 * u], [ex + .6 * u, ey + .2 * u]], sw * 1.1, PAL.ink, 'ink', .7);
      else {
        const big = f.eyes === 'wide' ? 1.15 : 1;
        paint(ellPts(ex, ey, .8 * u * big, 1.0 * u * big, 18, .15), { wash: '#FFFFFF', ink: PAL.ink, sw });
        if (f.eyes === 'star') paint(starPts(ex, ey, .55 * u, .45, 5, .2), { wash: '#FFD27A', ink: PAL.ink, sw: sw * .6 });
        else paint(ellPts(ex + lx * u, ey + ly * u - .05 * u, .3 * u * (big > 1 ? .8 : 1), .34 * u * (big > 1 ? .8 : 1), 12, .08), { wash: PAL.ink, ink: null });
        if (f.eyes === 'narrow') paint(rectPts(ex - .85 * u, ey - 1.05 * u, 1.7 * u, .75 * u, 0), { wash: col(M.white), ink: null }), inkLine([[ex - .8 * u, ey - .3 * u], [ex + .8 * u, ey - .3 * u + s * .15 * u]], sw, PAL.ink, 'ink', 0);
      }
    }
    const mx = .35 * u, my = -7.0 * u;
    if (f.mouth === 'smile') inkLine([[mx - .7 * u, my - .15 * u], [mx, my + .25 * u], [mx + .6 * u, my - .2 * u]], sw, PAL.ink, 'ink', .7);
    else if (f.mouth === 'grin' || f.mouth === 'open') paint([[mx - .8 * u, my - .2 * u], [mx + .75 * u, my - .25 * u], [mx, my + (f.mouth === 'open' ? .75 : .45) * u]], { wash: '#B8413A', ink: PAL.ink, sw, curv: .5 });
    else if (f.mouth === 'o') paint(ellPts(mx, my, .3 * u, .38 * u, 12, .1), { wash: '#B8413A', ink: PAL.ink, sw: sw * .8 });
    else inkLine([[mx - .5 * u, my], [mx + .5 * u, my]], sw, PAL.ink, 'ink', 0);
    pop();
    const em = o.emote === undefined ? f.emote : o.emote;
    if (em && (o.emoteK ?? 1) > 0) emote(em, x + (o.flip ? -3.2 : 3.2) * u, y + ((o.dy || 0) - 13) * u, u * .9, o.emoteK ?? 1, o.emoteAge ?? T);
  }

  // ---------- HOOK (0 → B): frame 0 is the thumbnail — the payoff, fully drawn, no fade-in ----------
  function shotHook(t, lt, dur) {
    camBegin(540, 520, lerp(1.0, 1.06, ease(seg(lt, 0, dur))));
    room(t, 1);
    for (let i = 0; i < 3; i++) card(i, t, 1, 12);
    brain(540, 300, .6, t, 0, 1);
    glow(540, 300, 280, C.gold, .55);
    for (let s = 0; s < 14; s++) {   // gold beads rising off the rig
      const q = frac(lt * .55 + hash(s)), x = CARDS[s % 3][0] + (hash(s + 5) - .5) * 160;
      bead(x, lerp(450, 170, q), 7, 1 - q);
    }
    mitu(540, FLOOR, 24, { ...feelM('starstruck', t), view: 'front', hat: 'hard', aL: 1.2, aR: 1.25 + .15 * Math.sin(lt * 9), boilKey: 'clawdHook' });
    camEnd();
    // headline: already at full size on frame 0 (pop 1), with a small beat pulse after
    const pk = 1 + .04 * pulse(t, 6);
    letter('117 tok/s', W / 2, 118, 116 * pk, C.gold, { screen: true, stroke: C.ink, rot: -.03, pop: 1 });
    letter('MiMo-V2.6-Flash · 309B', W / 2, 212, 46, '#FFE3EA', { screen: true, stroke: C.ink, rot: -.02, pop: 1 });
    letter('on 3 old mining cards', W / 2, 1012, 50, PAL.cream, { screen: true, stroke: C.ink, rot: .015, pop: 1 });
    if (lt > dur - .32) { flushLetters(); brushWipe((lt - (dur - .32)) / .64, [C.goldDk, C.gold]); }
  }

  // ---------- A: the rig asleep ----------
  function shotA(t, lt, dur) {
    camBegin(540, 560, lerp(1.25, 1.08, ease(seg(lt, 0, dur))));
    room(t);
    for (let i = 0; i < 3; i++) card(i, t, 0, i === 1 ? .8 * seg(lt, dur * .55, dur) : 0);
    const mood = emotionsM(lt, [[0, 'sleepy'], [dur * .72, 'surprised', { lookX: .6, lookY: -.5 }]]);
    mitu(560, FLOOR, 22, { ...mood, view: 'q' });
    camEnd();
    if (lt < .6) iris(540, 505, lerp(0, 900, easeIn(lt / .6)));
  }

  // ---------- B: three cards wake, memory fills ----------
  function shotB(t, lt, dur) {
    camBegin(lerp(540, 560, ease(seg(lt, 0, dur))), 600, 1.02 + .02 * Math.sin(lt));
    room(t, .1);
    const taps = [.05, .86, 1.46];   // "three" "old" "mining"
    const walk = kf(lt, [[0, 180], [taps[0], 210], [taps[1], 530], [taps[2], 850], [dur, 860]], ease);
    for (let i = 0; i < 3; i++) {
      const on = seg(lt, taps[i], taps[i] + .4);
      card(i, t, on * .55, fanSpin(lt, taps[i], taps[i] + .8, 4), { chips: seg(lt, 2.66, 3.6) });
    }
    const hop = ring(lt, taps);
    const mood = emotionsM(lt, [[0, 'surprised'], [taps[0] + .1, 'happy']], { take: .6 });
    mitu(walk, FLOOR, 22, { ...mood, view: 'side', walk: lt * 9, aR: 1.1 + .4 * hop, sq: (mood.sq || 0) + .08 * hop });
    label('64 GB', 860, 425, 40, seg(lt, 2.66, 2.95), { color: C.paper, rot: .03 });
    camEnd();
    if (lt < .32) { flushLetters(); brushWipe(.5 + lt / .64, [C.goldDk, C.gold]); }
  }

  // ---------- C: the giant brain arrives, gets sliced in three ----------
  function shotC(t, lt, dur) {
    const chops = [1.48, 1.96, 2.32], land = 3.0;   // chop on "three" "now" "it's"
    const sh = lt > chops[0] && lt < chops[2] + .3 ? shakeXY(t, 5) : [0, 0];
    camBegin(540 + sh[0], kf(lt, [[0, 420], [dur * .4, 470], [dur, 520]], ease) + sh[1], kf(lt, [[0, .92], [dur * .45, .96], [dur, 1.02]], ease));
    room(t, .15);
    const drop = easeOut(seg(lt, 0, .94));
    const slice = seg(lt, chops[2], land);
    for (let i = 0; i < 3; i++) card(i, t, lt > land ? seg(lt, land, land + .3) * .7 : .45, 4);
    if (slice < 1) brain(540, lerp(-200, 250, drop) + slice * 200, 1.25 - .5 * slice, t, slice);
    else for (let i = 0; i < 3; i++) {   // the three slices have landed on their cards
      const [cx, cy] = CARDS[i]; boilSeed('slab' + i);
      paint(ellPts(cx, cy - 78, 100, 30, 18, 1.5), { wash: C.brain, fill: C.brainDk, fillOp: 70, ink: C.ink, sw: .8 });
    }
    label('MiMo', 540, lerp(-200, 250, drop) - 150, 64, seg(lt, .5, .8) * (1 - seg(lt, chops[0], chops[0] + .3)), { color: '#FFE3EA' });
    for (const c of chops) { const k = seg(lt, c, c + .35); for (let s = 0; s < 4; s++) sparkle(540 + (s - 1.5) * 90, 330, 18, k); }
    const mood = emotionsM(lt, [[0, 'scared', { lookY: -1 }], [1.2, 'determined']]);
    const swing = ring(lt, chops);
    mitu(540, FLOOR, 22, { ...mood, hat: 'hard', aR: 1.4 - 1.2 * swing, view: 'front',
      armR: (u, sw) => { paint([[0, -u * .25], [u * .9, -u * .25], [u * .9, u * .25], [0, u * .25]], { wash: '#7A5C44', ink: C.ink, sw }); paint([[u * .8, -u * 1.3], [u * 3.8, -u * 1.5], [u * 3.9, u * .9], [u * .8, u * .8]], { wash: '#D5DADF', fill: '#AEB6BE', fillOp: 70, ink: C.ink, sw }); } });
    camEnd();
    if (lt < .45) { const k = lt / .45; paint(rectPts(-10, -10, W + 20, H * (1 - easeOut(k)) + 10, 0), { wash: C.wall, washOp: 200, ink: null }); }   // shadow sweeps off
  }

  // ---------- D: pipeline relay ----------
  function shotD(t, lt, dur) {
    camBegin(540, 520, 1.05 + .02 * Math.sin(lt * 1.3));
    room(t, .3);
    for (let i = 0; i < 3; i++) { const hit = pulse(t - i * BEAT * .66, 5); card(i, t, .6 + .35 * hit, 6); }
    for (let i = 0; i < 3; i++) { boilSeed('slabD' + i); paint(ellPts(CARDS[i][0], CARDS[i][1] - 78, 100, 30, 18, 1.5), { wash: C.brain, fill: C.brainDk, fillOp: 70, ink: C.ink, sw: .8 }); }
    const travel = BEAT * 2;
    for (let b = 0; b < 12; b++) {   // one bead per beat, each crossing the row in two beats
      const p = (lt - b * BEAT * .9) / travel; if (p < 0 || p > 1.15) continue;
      const [x, y] = relayPt(p); bead(x, y, 16, p > 1 ? 1 - (p - 1) / .15 : Math.min(1, p * 6));
    }
    const mood = feelM('proud', t);
    const baton = Math.sin(bpOf(t) * Math.PI);
    mitu(540, FLOOR, 22, { ...mood, view: 'front', aR: .6 + .9 * baton, aL: .2 - .5 * baton, hat: 'hard' });
    camEnd();
    if (lt < .35) flash(.45 * (1 - lt / .35), C.gold);
  }

  // ---------- E: the helper guesses two ahead ----------
  function shotE(t, lt, dur) {
    camBegin(560, 520, 1.05);
    room(t, .45);
    for (let i = 0; i < 3; i++) { card(i, t, .8, 9); boilSeed('slabE' + i); paint(ellPts(CARDS[i][0], CARDS[i][1] - 78, 100, 30, 18, 1.5), { wash: C.brain, fill: C.brainDk, fillOp: 70, ink: C.ink, sw: .8 }); }
    const speed = lerp(1, 2.2, ease(seg(lt, dur * .5, dur * .8)));
    for (let b = 0; b < 16; b++) {
      const p = (lt * speed - b * BEAT * .7) / (BEAT * 2); if (p < 0 || p > 1.15) continue;
      const [x, y] = relayPt(p); bead(x, y, 16, p > 1 ? 1 - (p - 1) / .15 : Math.min(1, p * 6));
      if (speed > 1.5) inkLine([[x - 40, y], [x - 12, y]], .6, C.gold, 'dry', 0);
    }
    // the helper: a tiny gold Clawd zipping in from the right, dropping two guessed beads ahead of the stream
    const hx = kf(lt, [[0, 1150], [.3, 780], [dur * .5, 720], [dur, 700]], easeOut);
    const guess = [.35, .65];
    for (const [j, g] of guess.entries()) {
      const k = seg(lt, g, g + .35), acc = seg(lt, 1.2 + (g - .35), 1.6 + (g - .35));
      const gx = hx - 90 - j * 60, gy = 400;
      if (k > 0) bead(gx, lerp(430, gy, easeOut(k)), 10, Math.min(1, k * 3));
      sparkle(gx, gy - 30, 22, acc);
    }
    mitu(hx, 440, 7, { ...feelM('excited', t), tint: 'gold', tintK: .7, view: 'side', flip: true, walk: lt * 12, noShadow: true, boilKey: 'helper' });
    const mood = emotionsM(lt, [[0, 'proud'], [dur * .25, 'surprised', { lookX: .8, lookY: -.6 }], [dur * .55, 'excited']]);
    mitu(430, FLOOR, 22, { ...mood, view: 'q', hat: 'hard', boilKey: 'clawdE' });
    camEnd();
    if (lt > dur - .3) { flushLetters(); brushWipe((lt - (dur - .3)) / .6, [C.goldDk, C.gold]); }
  }

  // ---------- F: decode 117 tok/s ----------
  function gauge(cx, cy, r, v, t) {
    boilSeed('gauge');
    paint(ellPts(cx, cy, r, r, 34, 1.5), { wash: PAL.cream, fill: '#EBDDBE', fillOp: 90, ink: C.ink, sw: 1.1 });
    for (let k = 0; k <= 15; k++) {   // ticks 0..150
      const a = Math.PI * (1 + k / 15), rr = k % 5 ? .84 : .76;
      inkLine([[cx + Math.cos(a) * r * rr, cy + Math.sin(a) * r * rr], [cx + Math.cos(a) * r * .93, cy + Math.sin(a) * r * .93]], k % 5 ? .6 : 1.1, C.ink, 'ink', 0);
    }
    paint([[cx - r * .93, cy], [cx + r * .93, cy], [cx + r * .93, cy + 10], [cx - r * .93, cy + 10]], { wash: PAL.cream, ink: null });
    const a = Math.PI * (1 + clamp(v / 150)) + .02 * Math.sin(t * 40) * (v > 100);
    glow(cx + Math.cos(a) * r * .6, cy + Math.sin(a) * r * .6, 60, C.gold, v / 150);
    paint(ribbon([[cx, cy], [cx + Math.cos(a) * r * .85, cy + Math.sin(a) * r * .85]], 12, 2), { wash: '#D9534A', ink: C.ink, sw: .6 });
    paint(ellPts(cx, cy, 16, 16, 12, .5), { wash: C.ink, ink: null });
  }
  function shotF(t, lt, dur) {
    const hit = .82;   // "117"
    const sh = lt > hit && lt < hit + .5 ? shakeXY(t, 6 * (1 - seg(lt, hit, hit + .5))) : [0, 0];
    camBegin(540 + sh[0], 440 + sh[1], lerp(1.0, 1.08, ease(seg(lt, 0, dur))));
    room(t, .55);
    for (let i = 0; i < 3; i++) card(i, t, .9, 14);
    const v = kf(lt, [[0, 0], [hit - .1, 117]], ease);
    gauge(540, 270, 140, v, t);
    label(Math.round(v) + "", 540, 245, 52, seg(lt, .05, .3), { color: C.ink, stroke: PAL.cream, rot: 0 });
    label('117 tok/s', 540, 110, 70, seg(lt, hit, hit + .4));
    label('decode · 1 stream', 540, 175, 36, seg(lt, 1.3, 1.6), { color: PAL.cream, rot: 0 });
    const mood = emotionsM(lt, [[0, 'excited'], [hit, 'starstruck']]);
    mitu(840, FLOOR, 22, { ...mood, ...move('bounce', t, 3), view: 'q', flip: true, hat: 'hard', boilKey: 'clawdF' });
    camEnd();
    if (lt < .3) { flushLetters(); brushWipe(.5 + lt / .6, [C.goldDk, C.gold]); }
    if (lt > dur - .3) { flushLetters(); brushWipe((lt - (dur - .3)) / .6, [C.pcbDk, C.pcb]); }
  }

  // ---------- G: prefill 4,000 tok/s ----------
  function shotG(t, lt, dur) {
    const gulp = .55;   // "4000" at .7
    camBegin(540, 520, 1.0);
    room(t, .6);
    for (let i = 0; i < 3; i++) card(i, t, .9 + .1 * pulse(t, 4), 14);
    // the prompt: a long paper scroll unrolling from the left, then sucked into the first card in one swoop
    const unroll = easeOut(seg(lt, 0, gulp)), suck = easeIn(seg(lt, gulp, gulp + .55));
    const x0 = lerp(-60, CARDS[0][0], suck), x1 = Math.max(x0 + 6, lerp(lerp(-40, 1000, unroll), CARDS[0][0] + 4, suck));
    const y0 = lerp(330, CARDS[0][1] - 40, suck);
    if (suck < .98) {
      boilSeed('scroll');
      const P = []; for (let k = 0; k <= 16; k++) { const x = lerp(x0, x1, k / 16); P.push([x, y0 + Math.sin(x * .015 + t * 3) * 16 * (1 - suck)]); }
      paint(ribbon(P, 80 * (1 - suck * .7), 80 * (1 - suck * .7)), { wash: PAL.cream, fill: '#E9DDC0', fillOp: 90, ink: C.ink, sw: .8 });
      for (let k = 1; k < 16; k += 1) { const [x, y] = P[k]; for (const dy of [-18, -2, 14]) inkLine([[x - 18, y + dy * (1 - suck)], [x + 14, y + dy * (1 - suck)]], .5, '#9C8C74', 'inkfine', 0); }
      if (x1 > 60) { boilSeed('roll'); paint(ellPts(x1, y0, 22, 44 * (1 - suck * .7), 14, .5), { wash: '#E9DDC0', ink: C.ink, sw: .7 }); }
    }
    if (lt > gulp) { const k = seg(lt, gulp + .3, gulp + .8); glow(CARDS[0][0], CARDS[0][1], 260 * (1 - k * .5), C.gold, 1 - k * .6); }
    label('4,000 tok/s', 540, 150, 70, seg(lt, .7, 1.0));
    label('prefill · 1 request', 540, 215, 36, seg(lt, 1.3, 1.6), { color: PAL.cream, rot: 0 });
    const mood = emotionsM(lt, [[0, 'surprised', { lookX: -.8 }], [gulp + .5, 'laugh']]);
    mitu(700, FLOOR, 22, { ...mood, view: 'q', flip: true, hat: 'hard', boilKey: 'clawdG' });
    camEnd();
    if (lt < .3) { flushLetters(); brushWipe(.5 + lt / .6, [C.pcbDk, C.pcb]); }
  }

  // ---------- H: 32 streams, 561 tok/s ----------
  function shotH(t, lt, dur) {
    camBegin(540, lerp(520, 470, ease(seg(lt, 0, dur))), lerp(1.08, .9, ease(seg(lt, 0, dur * .6))));
    room(t, .75);
    for (let i = 0; i < 3; i++) card(i, t, 1, 16);
    const grow = seg(lt, .1, 1.6);
    for (let s = 0; s < 32; s++) {   // a fountain: 32 streams arcing up to 32 little cups
      if (s / 32 > grow) continue;
      const from = CARDS[s % 3], a = Math.PI * (1.08 + .84 * s / 31);
      const to = [540 + Math.cos(a) * 470, 560 + Math.sin(a) * 440];
      boilSeed('cup' + s);
      paint(rrPts(to[0] - 12, to[1] - 10, 24, 18, 4, .4), { wash: PAL.cream, ink: C.ink, sw: .5 });
      for (let b = 0; b < 2; b++) {
        const p = frac(lt * .9 + hash(s) + b * .5), [x, y] = arcPt([from[0], from[1] - 60], to, 80, ease(p));
        if (s % 2 === 0 || b === 0) bead(x, y, 5, 1);
      }
    }
    label('32 streams', 540, 75, 50, seg(lt, .05, .35), { color: PAL.cream });
    label('561 tok/s', 540, 180, 80, seg(lt, 2.34, 2.64));
    const mood = emotionsM(lt, [[0, 'excited'], [2.34, 'starstruck']]);
    mitu(540, FLOOR, 22, { ...mood, view: 'front', hat: 'hard', boilKey: 'clawdH' });
    camEnd();
  }

  // ---------- I: the rhyme — rig glowing gold, brain as lantern, Clawd on top ----------
  function shotI(t, lt, dur) {
    camBegin(540, 470, lerp(.95, 1.05, ease(seg(lt, 0, dur))));
    room(t, 1);
    for (let i = 0; i < 3; i++) card(i, t, 1, 10);
    brain(540, 250, .55, t, 0, 1);
    glow(540, 250, 260, C.gold, .6);
    // streams swirling back into the cards at the start of the shot
    const sw = 1 - seg(lt, 0, dur * .3);
    for (let s = 0; s < 18 * sw; s++) { const a = s * .7 + lt * 5, r = 300 * sw * (.4 + .6 * hash(s)); bead(540 + Math.cos(a) * r, 470 + Math.sin(a) * r * .7, 5, sw); }
    const up = seg(lt, .3, 1.3);
    const hop = jump(lt, .3, 1.1, 6);
    mitu(lerp(760, 540, up), lerp(FLOOR, 450, easeOut(up)), 13, { ...feelM('proud', t), dy: hop.dy, sq: hop.sq, view: 'front', hat: 'hard', boilKey: 'clawdI' });
    for (let k = 0; k < 14; k++) {   // "watch it go": a burst of sparks off the rig
      const q = seg(lt, 2.2 + hash(k) * .3, 3.2 + hash(k) * .3), ang = -Math.PI * (.1 + .8 * hash(k + 9));
      sparkle(540 + Math.cos(ang) * 420 * q, 470 + Math.sin(ang) * 380 * q, 24, q);
    }
    label('MiMo-V2.6-Flash', 540, 700, 54, seg(lt, .6, .9), { color: PAL.cream });
    label('3× CMP 170HX', 540, 775, 48, seg(lt, 1.26, 1.56));
    camEnd();
    if (lt > dur - .7) iris(540, 400, lerp(900, 0, easeIn(seg(lt, dur - .7, dur))));
  }


  // ---------- lyrics: karaoke subtitles along the bottom, word times from take mimo_synthpop_s11 (video time) ----------
  const LYRICS = [
    [[1.92, 'Three'], [2.78, 'old'], [3.38, 'mining'], [3.86, 'cards,'], [4.58, 'sixty-four'], [5.06, 'gigs'], [5.88, 'each']],
    [[6.36, 'Slice'], [6.66, 'the'], [6.84, 'giant'], [7.30, 'brain'], [7.66, 'in'], [7.84, 'three,'], [8.34, 'now'], [8.68, "it's"], [9.34, 'in'], [9.70, 'reach']],
    [[10.24, 'Pipeline,'], [11.34, 'pipeline,'], [11.88, 'tokens'], [12.72, 'down'], [12.98, 'the'], [13.16, 'line']],
    [[13.60, 'Guess'], [13.92, 'two'], [14.14, 'tokens'], [14.80, 'ahead,'], [15.68, 'every'], [16.04, 'layer'], [16.50, 'right'], [16.92, 'on'], [17.10, 'time']],
    [[18.00, 'Decode'], [18.82, 'one-seventeen,'], [20.26, 'tokens'], [20.96, 'every'], [21.34, 'second']],
    [[21.96, 'Prefill'], [22.66, 'four thousand,'], [23.98, 'faster'], [24.54, 'than'], [25.18, 'you'], [25.50, 'reckoned']],
    [[26.22, 'Thirty-two'], [27.08, 'streams,'], [28.56, 'five-sixty-one'], [29.14, 'flow']],
    [[29.80, 'MiMo'], [30.50, 'on'], [30.84, 'the'], [31.06, 'one-seventy,'], [32.00, 'watch'], [32.86, 'it'], [33.10, 'go']],
  ];
  const LYR_SIZE = 40, LYR_MAXW = 1020, LYR_Y = 1016;
  const MCTX = document.createElement('canvas').getContext('2d');
  const tw = (txt, size) => { MCTX.font = `${size}px "Permanent Marker", "Comic Sans MS", cursive`; return MCTX.measureText(txt).width; };
  function lyrics(t) {
    let li = -1; for (let i = 0; i < LYRICS.length; i++) if (t >= LYRICS[i][0][0] - .2) li = i;
    if (li < 0) return;
    const line = LYRICS[li], next = LYRICS[li + 1], t0 = line[0][0] - .2, t1 = next ? next[0][0] - .2 : 34.6;
    if (t > t1 + .15) return;
    const inK = seg(t, t0, t0 + .25), outK = 1 - seg(t, t1 - .12, t1 + .1);
    const sp = tw(' ', LYR_SIZE) + 10, words = line.map(wd => [...wd, tw(wd[1], LYR_SIZE)]);
    const width = ws => ws.reduce((a, wd) => a + wd[2], 0) + sp * Math.max(0, ws.length - 1);
    let rows = [words];
    if (width(words) > LYR_MAXW) {   // two balanced rows: pick the split that minimises the wider row
      let best = 1, bestW = Infinity;
      for (let k = 1; k < words.length; k++) { const m = Math.max(width(words.slice(0, k)), width(words.slice(k))); if (m < bestW) { bestW = m; best = k; } }
      rows = [words.slice(0, best), words.slice(best)];
    }
    rows.forEach((row, r) => {
      const total = row.reduce((a, wd) => a + wd[2], 0) + sp * (row.length - 1);
      let x = W / 2 - total / 2; const y = LYR_Y + (r - (rows.length - 1) / 2) * LYR_SIZE * 1.15;
      for (const [ws, txt, ww] of row) {
        const sung = seg(t, ws - .04, ws + .08), bump = 1 + .07 * Math.sin(Math.PI * seg(t, ws - .04, ws + .22));
        letter(txt, x + ww / 2, y - 8 * (1 - inK), LYR_SIZE * bump, mixCol('#F3E9D2', C.gold, sung),
          { screen: true, stroke: C.ink, rot: 0, alpha: inK * outK, pop: .6 + .4 * inK });
        x += ww + sp;
      }
    });
  }

  const withLyrics = fn => (t, lt, dur) => { fn(t, lt, dur); lyrics(t); };
  shots([[START.A, withLyrics(shotHook)], [START.B, withLyrics(shotB)], [START.C, withLyrics(shotC)], [START.D, withLyrics(shotD)], [START.E, withLyrics(shotE)],
         [START.F, withLyrics(shotF)], [START.G, withLyrics(shotG)], [START.H, withLyrics(shotH)], [START.I, withLyrics(shotI)]]);
})();
