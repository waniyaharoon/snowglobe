
import sys
import time
import numpy as np

rng = np.random.default_rng(7)

# ------------------------------------------------------------------ 1. SETTINGS
CW, CH = 5, 6                  # one character cell = 5 x 6 pixels
GW, GH = 160, 100              # grid of cells  -> window is 800 x 600
W, H = GW * CW, GH * CH
FOCAL = GW * 0.9               # focal length in cells (zoom)
ASPECT = CW / CH               # cells are taller than wide, correct for it
NEAR = 0.05                    # things closer than this to the camera are dropped
T_LOOP = 14.5                  # seconds for the whole camera move


def rgb(h):
    """0xRRGGBB -> array of three floats 0..1"""
    return np.array([(h >> 16) & 255, (h >> 8) & 255, h & 255]) / 255.0


SNOW = rgb(0xEEF3FF)
WOOD = rgb(0x8A5230)
GREEN = rgb(0x1F7A3A)
COAL = rgb(0x15151A)
CARROT = rgb(0xFF7A1A)
BLUE = rgb(0x2F6BFF)
RED = rgb(0xFF3B3B)
LAMPCOL = rgb(0xFFD36B)


# ------------------------------------------------------------------ 2. PIECES
# A "piece" is a list of 5 arrays, all with one row per point:
#   P (n,3) position   N (n,3) surface normal   alb (n,3) base colour
#   sp (n,) spacing between neighbouring points (so we know how big to draw each dot)
#   emit (n,) 0..1, how much the point glows by itself (lights, lamp, snow)
def piece(P, N, alb, sp, emit=0.0):
    n = len(P)
    alb = np.broadcast_to(np.asarray(alb, float), (n, 3)).copy()
    sp = np.broadcast_to(np.asarray(sp, float), (n,)).copy()
    emit = np.broadcast_to(np.asarray(emit, float), (n,)).copy()
    return [np.asarray(P, float), np.asarray(N, float), alb, sp, emit]


def cat(*ps):
    return [np.concatenate([p[i] for p in ps]) for i in range(5)]


def move(p, offset):
    p[0] = p[0] + np.asarray(offset, float)
    return p


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


# ------------------------------------------------------------------ 3. SHAPE MAKERS
def sphere(c, r, n, alb, emit=0.0):
    # random normal-distributed vectors, normalised = uniform points on a sphere
    N = unit(rng.normal(size=(n, 3)))
    return piece(np.asarray(c, float) + N * r, N, alb, np.sqrt(4 * np.pi * r * r / n), emit)


def basis(axis):
    """given an axis, return (axis, u, v): three mutually perpendicular unit vectors"""
    a = unit(np.asarray(axis, float))
    h = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = unit(np.cross(a, h))
    v = np.cross(a, u)
    return a, u, v


def tube(p0, p1, r, n, alb, emit=0.0):
    """a cylinder surface from p0 to p1"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    a, u, v = basis(p1 - p0)
    L = np.linalg.norm(p1 - p0)
    t = rng.random(n) * L
    th = rng.random(n) * 2 * np.pi
    N = np.cos(th)[:, None] * u + np.sin(th)[:, None] * v
    P = p0 + a * t[:, None] + N * r
    return piece(P, N, alb, np.sqrt(2 * np.pi * r * L / n), emit)


def torus(c, R, r, n, alb):
    """a donut lying flat (used for the scarf)"""
    a = rng.random(n) * 2 * np.pi
    b = rng.random(n) * 2 * np.pi
    ring = np.stack([np.cos(a), np.zeros(n), np.sin(a)], 1)
    N = np.cos(b)[:, None] * ring + np.sin(b)[:, None] * np.array([0, 1.0, 0])
    P = np.asarray(c, float) + R * ring + r * N
    return piece(P, N, alb, np.sqrt(4 * np.pi ** 2 * R * r / n), 0.0)


def revolve(profile, n, alb, emit=0.0, side=None):
    """
    Surface of revolution (like a lathe / pottery wheel).
    profile = [(radius, height), ...] polyline, spun around the Y axis.
    Used for the base, ground, tree cones, hats.
    side=+1/-1 forces which way normals point; None = away from profile centre.
    """
    pr = np.asarray(profile, float)
    a, b = pr[:-1], pr[1:]
    d = b - a
    seg_len = np.linalg.norm(d, axis=1)
    area = np.maximum(seg_len * (a[:, 0] + b[:, 0]) / 2 * 2 * np.pi, 1e-9)
    k = rng.choice(len(a), size=n, p=area / area.sum())     # more points on bigger segments
    u = rng.random(n)
    r0, r1 = a[k, 0], b[k, 0]
    dr = r1 - r0
    with np.errstate(all="ignore"):                          # uniform-by-area along a cone
        t = np.where(np.abs(dr) < 1e-9, u, (-r0 + np.sqrt(r0 ** 2 + u * (r1 ** 2 - r0 ** 2))) / dr)
    t = np.nan_to_num(t)
    rr, yy = r0 + dr * t, a[k, 1] + d[k, 1] * t
    th = rng.random(n) * 2 * np.pi
    n2 = np.stack([d[k, 1], -d[k, 0]], 1) / seg_len[k, None]  # 2D normal of each profile segment
    if side is None:
        cen = pr.mean(0)
        flip = np.sign(np.sum(n2 * (np.stack([rr, yy], 1) - cen), 1))
        flip[flip == 0] = 1
    else:
        flip = side
    n2 = n2 * (flip[:, None] if np.ndim(flip) else flip)
    P = np.stack([rr * np.cos(th), yy, rr * np.sin(th)], 1)
    N = np.stack([n2[:, 0] * np.cos(th), n2[:, 1], n2[:, 0] * np.sin(th)], 1)
    return piece(P, N, alb, np.sqrt(area.sum() / n) * 1.1, emit)


# ------------------------------------------------------------------ 4. LIGHTING
SUN = unit(np.array([-0.5, 0.8, 0.6]))        # direction pointing TO the moonlight
LAMP_POS = np.array([-0.45, 0.60, -0.12])


def shade(P, N, alb, emit):
    """Lambert shading: a surface is brighter the more it faces the light."""
    moon = np.clip(N @ SUN, 0, 1)[:, None] * np.array([0.55, 0.62, 0.85])
    to = LAMP_POS - P
    d2 = np.sum(to ** 2, 1) + 1e-6
    to = to / np.sqrt(d2)[:, None]
    lamp = (np.clip(np.sum(N * to, 1), 0, 1) * 0.12 / (d2 + 0.05))[:, None] * np.array([1.0, 0.72, 0.38])
    ambient = np.array([0.16, 0.19, 0.30])
    lit = alb * (ambient + moon + lamp)
    e = emit[:, None]
    return np.clip(lit * (1 - e) + alb * e, 0, 1.4)


# ------------------------------------------------------------------ 5. THE SCENE
DOME_C = np.array([0.0, 0.55, 0.0])
DOME_R = 1.0


def build_static():
    parts = []
    # wooden pedestal (revolve a side profile)
    parts.append(revolve([(0, -0.5), (1.05, -0.5), (1.05, -0.4), (0.95, -0.25), (0.88, -0.05), (0.88, 0.0)],
                         14000, WOOD * rng.uniform(0.8, 1.1, (14000, 1))))
    # snowy ground: top is a flat disc, normals up
    g = revolve([(0.0, 0.03), (0.5, 0.02), (0.88, 0.0)], 30000, SNOW, side=-1)
    g[2] = g[2] * rng.uniform(0.88, 1.0, (len(g[0]), 1))
    parts.append(g)
    # snow mounds
    for _ in range(14):
        a, rad = rng.random() * 2 * np.pi, 0.75 * np.sqrt(rng.random())
        m = sphere((rad * np.cos(a), 0.0, rad * np.sin(a)), rng.uniform(0.03, 0.09), 1200, SNOW)
        keep = m[0][:, 1] > 0.0
        parts.append([x[keep] for x in m])

    # tree
    tree = [tube((0, 0, 0), (0, 0.14, 0), 0.03, 1200, rgb(0x5A3A22))]
    for y0, y1, r in [(0.10, 0.36, 0.23), (0.26, 0.52, 0.18), (0.42, 0.65, 0.13), (0.55, 0.80, 0.08)]:
        cone = revolve([(0, y0), (r, y0), (0, y1)], 4500, GREEN * rng.uniform(0.75, 1.1, (4500, 1)))
        on_top = (cone[1][:, 1] > 0.55) & (rng.random(len(cone[0])) < 0.65)       # snow on upward-facing parts
        cone[2][on_top] = SNOW
        bulbs = rng.random(len(cone[0])) < 0.012                                  # glowing lights
        palette = np.array([RED, LAMPCOL, BLUE, rgb(0xFF5CE0)])
        cone[2][bulbs] = palette[rng.integers(0, 4, bulbs.sum())]
        cone[4][bulbs] = 1.0
        tree.append(cone)
    tree.append(sphere((0, 0.82, 0), 0.025, 500, LAMPCOL, emit=1.0))
    parts.append(move(cat(*tree), (0.48, 0.0, -0.18)))

    # lamp post
    lamp = [tube((0, 0, 0), (0, 0.5, 0), 0.012, 900, rgb(0x15161C)),
            tube((0, 0, 0), (0, 0.05, 0), 0.025, 500, rgb(0x15161C)),
            tube((0, 0.50, 0), (0, 0.62, 0), 0.04, 1500, LAMPCOL, emit=1.0),
            revolve([(0.05, 0.62), (0.0, 0.68)], 400, rgb(0x15161C), side=1)]
    parts.append(move(cat(*lamp), (LAMP_POS[0], 0.0, LAMP_POS[2])))

    P, N, alb, sp, emit = cat(*parts)
    return [P, shade(P, N, alb, emit), sp]


def make_snowman(scarf_col, hat_col, s):
    """built facing +Z, standing on y=0 at the origin; s = size scale"""
    body = sphere((0, 0.10 * s, 0), 0.10 * s, 5200, SNOW)
    head = sphere((0, 0.245 * s, 0), 0.065 * s, 3200, SNOW)
    for ang in (-0.35, 0.35):                       # eyes: colour the points facing a direction
        e = unit(np.array([ang, 0.25, 0.9]))
        head[2][head[1] @ e > 0.985] = COAL
    for y in (0.3, 0.0, -0.3):                      # coal buttons
        b = unit(np.array([0, y, 0.95]))
        body[2][body[1] @ b > 0.992] = COAL
    nose = tube((0, 0.24 * s, 0.06 * s), (0, 0.235 * s, 0.14 * s), 0.011 * s, 500, CARROT)
    scarf = torus((0, 0.19 * s, 0), 0.062 * s, 0.018 * s, 1800, scarf_col)
    tail = tube((0.04 * s, 0.19 * s, 0.055 * s), (0.05 * s, 0.10 * s, 0.098 * s), 0.012 * s, 500, scarf_col)
    hat = [revolve([(0.03 * s, 0.305 * s), (0.085 * s, 0.305 * s)], 500, COAL, side=-1),   # brim top
           revolve([(0.03 * s, 0.305 * s), (0.085 * s, 0.305 * s)], 300, COAL, side=1),    # brim bottom
           revolve([(0.05 * s, 0.305 * s), (0.05 * s, 0.375 * s)], 700, hat_col, side=1),  # crown side
           revolve([(0.0, 0.375 * s), (0.05 * s, 0.375 * s)], 300, hat_col, side=-1)]      # crown top
    return cat(body, head, nose, scarf, tail, *hat)


SNOWMAN_A = make_snowman(BLUE, rgb(0x1E3FA8), 1.0)
SNOWMAN_B = make_snowman(RED, rgb(0xB01E2E), 0.82)
PAIR_CENTRE = np.array([0.0, 0.0, 0.12])


def place(local, pos, yaw, bob):
    """rotate a piece about Y by yaw, then move it to pos (+ a hop)"""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    P, N, alb, sp, emit = local
    return [P @ R.T + pos + np.array([0, bob, 0]), N @ R.T, alb, sp, emit]


def dancers(t):
    """the two snowmen waltz around each other"""
    a = 2 * np.pi * 5 * t / T_LOOP                       # 5 turns per loop, so it repeats cleanly
    off = 0.11 * np.array([np.cos(a), 0, np.sin(a)])
    out = []
    for local, sign, extra in ((SNOWMAN_A, 1, 0.0), (SNOWMAN_B, -1, np.pi)):
        pos = PAIR_CENTRE + sign * off
        face = -sign * off                                # look at the partner
        yaw = np.arctan2(face[0], face[2]) + 0.35 * np.sin(2 * a)
        p = place(local, pos, yaw, 0.012 * abs(np.sin(2 * a + extra)))
        out.append(p)
    P, N, alb, sp, emit = cat(*out)
    return P, shade(P, N, alb, emit), sp


# falling snow: position is a pure function of time t (no stored state)
_n = 2200
_xz = rng.uniform(-0.95, 0.95, (_n, 2))
_xz = _xz[np.linalg.norm(_xz, axis=1) < 0.95]
SN_N = len(_xz)
SN_X0, SN_Z0 = _xz[:, 0], _xz[:, 1]
SN_Y0 = rng.random(SN_N) * 1.5
SN_V = 0.10 + 0.12 * rng.random(SN_N)
SN_PH = rng.random(SN_N) * 6.28


def snow(t):
    y = np.mod(SN_Y0 - SN_V * t, 1.5)
    x = SN_X0 + 0.04 * np.sin(1.7 * t + SN_PH)
    z = SN_Z0 + 0.04 * np.cos(1.3 * t + SN_PH)
    P = np.stack([x, y, z], 1)
    inside = np.sum((P - DOME_C) ** 2, 1) < (DOME_R - 0.07) ** 2
    P = P[inside]
    return P, np.full((len(P), 3), 0.97), np.full(len(P), 0.005)


# glass dome: points on the upper sphere, coloured every frame by view angle
_G = unit(rng.normal(size=(24000, 3)))
_G = _G[_G[:, 1] * DOME_R + DOME_C[1] > 0.02]
GLASS_N = _G
GLASS_P = DOME_C + _G * DOME_R
GLASS_SP = np.sqrt(4 * np.pi * DOME_R ** 2 / 24000)

STATIC = build_static()


# ------------------------------------------------------------------ 6. CAMERA
def smooth(x):
    return x * x * (3 - 2 * x)


def camera(t):
    u = (t % T_LOOP) / T_LOOP
    c = 0.5 - 0.5 * np.cos(2 * np.pi * u)               # 0 -> 1 -> 0 over the loop
    e = smooth(np.clip((c - 0.12) / 0.7, 0, 1))         # clamp = hold at both ends, smooth in between
    far_t = np.array([0.0, 0.40, 0.0])
    near_t = np.array([0.0, 0.20, 0.12])
    tgt = far_t + (near_t - far_t) * e
    dist = 3.4 + (0.78 - 3.4) * e
    yaw = 0.45 + 0.9 * np.sin(2 * np.pi * u)
    pitch = 0.26 - 0.17 * e
    cam = tgt + dist * np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
    fwd = unit(tgt - cam)
    right = unit(np.cross(fwd, np.array([0, 1.0, 0])))
    up = np.cross(right, fwd)
    return cam, np.stack([right, up, fwd])             # rows of R = camera axes


# ------------------------------------------------------------------ 7. RENDERER
def project(P, sp, cam, R):
    """
    world points -> (cell index, depth, source index).
    1) view transform: express points in camera axes (dot with right/up/forward)
    2) perspective: cell = centre + focal * x / z
    3) splat: a point bigger than a cell covers s x s cells
    """
    rel = P - cam
    x, y, z = rel @ R[0], rel @ R[1], rel @ R[2]
    ok = np.nonzero(z > NEAR)[0]
    x, y, z, spk = x[ok], y[ok], z[ok], sp[ok]
    cx = GW / 2 + FOCAL * x / z
    cy = GH / 2 - FOCAL * ASPECT * y / z
    size = np.clip(np.ceil(spk * FOCAL / z), 1, 4).astype(int)
    L, Z, S = [], [], []
    for s in (1, 2, 3, 4):
        sel = np.nonzero(size == s)[0]
        if sel.size == 0:
            continue
        x0 = np.floor(cx[sel] - (s - 1) / 2).astype(int)
        y0 = np.floor(cy[sel] - (s - 1) / 2).astype(int)
        for dx in range(s):
            for dy in range(s):
                xx, yy = x0 + dx, y0 + dy
                m = (xx >= 0) & (xx < GW) & (yy >= 0) & (yy < GH)
                L.append(yy[m] * GW + xx[m])
                Z.append(z[sel][m])
                S.append(ok[sel][m])
    return np.concatenate(L), np.concatenate(Z), np.concatenate(S)


# ------------------------------------------------------------------ 8. ASCII LOOK
LEVELS = 14
_rank = rng.permutation(CH * CW).reshape(CH, CW)
# glyph k lights the first k/LEVELS of the pixels in a fixed random order: denser = brighter
ATLAS = np.stack([(_rank < (k / LEVELS) * CH * CW) for k in range(LEVELS + 1)])   # (LEVELS+1, CH, CW)


def to_ascii(fb):
    """fb: (GH,GW,3) floats -> (H,W,3) uint8 image made of coloured glyphs"""
    lum = np.clip(fb.max(axis=2), 0, 1)
    lev = np.round(lum ** 0.7 * LEVELS).astype(int)
    hue = fb / np.maximum(lum[..., None], 1e-6)               # colour at full strength
    hue = np.clip(hue, 0, 1)
    tiles = ATLAS[lev]                                         # (GH,GW,CH,CW) bool
    img = tiles[..., None] * hue[:, :, None, None, :]          # tint each glyph
    img = img.transpose(0, 2, 1, 3, 4).reshape(GH * CH, GW * CW, 3)
    return (img * 255).astype(np.uint8)


def render(t):
    cam, R = camera(t)
    P_s, C_s, S_s = STATIC
    P_d, C_d, S_d = dancers(t)
    P_n, C_n, S_n = snow(t)
    P = np.concatenate([P_s, P_d, P_n])
    C = np.concatenate([C_s, C_d, C_n])
    S = np.concatenate([S_s, S_d, S_n])

    # z-buffer: draw far -> near, the last write to a cell (nearest) wins
    lin, z, src = project(P, S, cam, R)
    order = np.argsort(-z)
    lin, z, src = lin[order], z[order], src[order]
    fb = np.zeros((GH * GW, 3))
    zb = np.full(GH * GW, np.inf)
    fb[lin] = C[src]
    zb[lin] = z

    # glass: blend a see-through layer where it is in front of what is already drawn
    v = unit(cam - GLASS_P)
    ndv = np.abs(np.sum(GLASS_N * v, 1))
    rim = (1 - ndv) ** 3                                       # bright where the surface is edge-on (Fresnel idea)
    spec = np.clip(np.sum(GLASS_N * unit(SUN + v), 1), 0, 1) ** 120
    alpha = np.clip(0.05 + 0.75 * rim + 0.6 * spec, 0, 1)
    gcol = np.clip(np.array([0.55, 0.75, 1.0]) * (0.55 + 0.45 * rim[:, None]) + spec[:, None], 0, 1)
    gl, gz, gs = project(GLASS_P, np.full(len(GLASS_P), GLASS_SP), cam, R)
    keep = gz < zb[gl]
    gl, gz, gs = gl[keep], gz[keep], gs[keep]
    go = np.argsort(-gz)
    gl, gs = gl[go], gs[go]
    base = fb[gl]
    fb[gl] = base * (1 - alpha[gs, None]) + gcol[gs] * alpha[gs, None]

    return to_ascii(fb.reshape(GH, GW, 3))


# ------------------------------------------------------------------ 9. MAIN LOOP
def main():
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    t0 = time.time()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
        img = render(time.time() - t0)                                    # (H, W, 3)
        pygame.surfarray.blit_array(screen, np.ascontiguousarray(img.transpose(1, 0, 2)))
        pygame.display.flip()
        clock.tick(30)
        pygame.display.set_caption(f"Snow Globe  {clock.get_fps():.0f} fps")
    pygame.quit()


if __name__ == "__main__":
    main()