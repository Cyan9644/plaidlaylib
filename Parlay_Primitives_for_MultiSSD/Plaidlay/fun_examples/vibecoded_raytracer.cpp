// raytracer.cpp
// A small reflective ray tracer. All scene/camera inputs are hard-coded.
// Outputs a binary P6 PPM image to stdout; status goes to stderr.
//
// Build (needs ParlayLib headers: https://github.com/cmuparlay/parlaylib):
//   g++ -std=c++17 -O2 -I ../../deps/parlaylib/ vibecoded_raytracer.cpp -o raytracer -pthread
// Run:
//   ./raytracer > out.ppm
//   (optional) PARLAY_NUM_THREADS=8 ./raytracer > out.ppm
//
// Structure / how the "primitives" are used:
//   * The image is a MAP over pixels  -> parlay::tabulate(W*H, ...)   (the real parallelism)
//   * A brightness stat is a REDUCE   -> parlay::reduce(parlay::map(...))
//   * The byte buffer is a MAP        -> parlay::tabulate(W*H*3, ...)
//   * Per ray, nearest-hit is "map over spheres -> keep front hits -> reduce to min",
//     and the shadow test is a "filter" (is anything between point and light?).
//     Those inner loops are kept serial on purpose: nesting parlay inside the hot
//     per-pixel loop over a tiny scene would only add overhead.

#include <parlay/primitives.h>
#include <parlay/sequence.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <iostream>
#include <algorithm>

// ----------------------------------------------------------------------------
// Vec3
// ----------------------------------------------------------------------------
struct Vec3 {
  float x = 0, y = 0, z = 0;
  Vec3() = default;
  Vec3(float a, float b, float c) : x(a), y(b), z(c) {}
  float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};

static inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline Vec3 operator-(Vec3 a)         { return {-a.x, -a.y, -a.z}; }
static inline Vec3 operator*(Vec3 a, float s){ return {a.x * s, a.y * s, a.z * s}; }
static inline Vec3 operator*(float s, Vec3 a){ return a * s; }
static inline Vec3 operator*(Vec3 a, Vec3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
static inline Vec3 operator/(Vec3 a, float s){ return {a.x / s, a.y / s, a.z / s}; }

static inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline float length(Vec3 a)      { return std::sqrt(dot(a, a)); }
static inline Vec3  normalize(Vec3 a)    { return a / length(a); }
static inline Vec3  cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static inline Vec3 reflect(Vec3 v, Vec3 n) { return v - n * (2.0f * dot(v, n)); }

// ----------------------------------------------------------------------------
// Scene types
// ----------------------------------------------------------------------------
struct Ray { Vec3 orig, dir; Vec3 at(float t) const { return orig + dir * t; } };

struct Sphere {
  Vec3  center;
  float radius;
  Vec3  albedo;
  float reflectivity;  // 0 = matte, 1 = mirror
  bool  checker;       // procedural checkerboard (used for the ground)
};

struct Hit { bool hit = false; float t = 0; Vec3 point, normal; int idx = -1; };

struct Light { Vec3 pos; Vec3 color; float intensity; };

// ----------------------------------------------------------------------------
// Shading helpers
// ----------------------------------------------------------------------------
static inline Vec3 skyColor(Vec3 dir) {
  float t = 0.5f * (normalize(dir).y + 1.0f);
  return Vec3(1, 1, 1) * (1.0f - t) + Vec3(0.5f, 0.7f, 1.0f) * t;
}

static inline Vec3 checkerColor(Vec3 p) {
  int c = static_cast<int>(std::floor(p.x * 0.6f) + std::floor(p.z * 0.6f));
  return ((c & 1) == 0) ? Vec3(0.85f, 0.85f, 0.85f) : Vec3(0.12f, 0.22f, 0.32f);
}

// Nearest intersection: scan spheres, keep hits in [tmin, best], reduce to closest.
static Hit closest(const Ray& r, const std::vector<Sphere>& scene, float tmin, float tmax) {
  Hit best; best.hit = false; best.t = tmax;
  for (int i = 0; i < static_cast<int>(scene.size()); ++i) {
    const Sphere& s = scene[i];
    Vec3 oc   = r.orig - s.center;
    float a   = dot(r.dir, r.dir);
    float halfb = dot(oc, r.dir);
    float c   = dot(oc, oc) - s.radius * s.radius;
    float disc = halfb * halfb - a * c;
    if (disc < 0.0f) continue;
    float sq = std::sqrt(disc);
    float t = (-halfb - sq) / a;
    if (t < tmin || t > best.t) {
      t = (-halfb + sq) / a;
      if (t < tmin || t > best.t) continue;
    }
    best.hit = true; best.t = t; best.idx = i;
  }
  if (best.hit) {
    best.point  = r.at(best.t);
    best.normal = normalize(best.point - scene[best.idx].center);
  }
  return best;
}

static Vec3 trace(const Ray& r, const std::vector<Sphere>& scene,
                  const Light& light, int depth) {
  if (depth <= 0) return Vec3(0, 0, 0);

  Hit h = closest(r, scene, 1e-3f, 1e30f);
  if (!h.hit) return skyColor(r.dir);

  const Sphere& s = scene[h.idx];
  Vec3 albedo = s.checker ? checkerColor(h.point) : s.albedo;
  Vec3 N = h.normal;
  Vec3 V = normalize(r.orig - h.point);  // toward camera

  Vec3 col = albedo * 0.12f;             // ambient fill

  // Direct light, gated by a shadow ray (the "filter": is the light visible?)
  Vec3 toL = light.pos - h.point;
  float dist = length(toL);
  Vec3 L = toL / dist;
  Ray shadow{ h.point + N * 1e-3f, L };
  Hit occ = closest(shadow, scene, 1e-3f, dist - 1e-3f);
  if (!occ.hit) {
    float diff = std::max(0.0f, dot(N, L));
    col = col + albedo * light.color * (diff * light.intensity);
    Vec3 Hh = normalize(L + V);
    float spec = std::pow(std::max(0.0f, dot(N, Hh)), 64.0f);
    col = col + light.color * (spec * 0.45f * light.intensity);
  }

  // Recursive reflection (bounded depth)
  if (s.reflectivity > 0.0f) {
    Ray rr{ h.point + N * 1e-3f, normalize(reflect(r.dir, N)) };
    Vec3 refl = trace(rr, scene, light, depth - 1);
    col = col * (1.0f - s.reflectivity) + refl * s.reflectivity;
  }
  return col;
}

// ----------------------------------------------------------------------------
// Camera
// ----------------------------------------------------------------------------
struct Camera {
  Vec3 pos, forward, right, up;
  float halfW, halfH;

  Camera(Vec3 from, Vec3 lookAt, Vec3 worldUp, float vfovDeg, float aspect) {
    pos     = from;
    forward = normalize(lookAt - from);
    right   = normalize(cross(forward, worldUp));
    up      = cross(right, forward);
    halfH   = std::tan(vfovDeg * 0.5f * 3.14159265358979f / 180.0f);
    halfW   = aspect * halfH;
  }
  // sx,sy in [0,1]; (0,0) = top-left.
  Ray ray(float sx, float sy) const {
    float u = (2.0f * sx - 1.0f) * halfW;
    float v = (1.0f - 2.0f * sy) * halfH;
    return Ray{ pos, normalize(forward + right * u + up * v) };
  }
};

// ----------------------------------------------------------------------------
// Beaded text: a full 8x8 ASCII bitmap font rendered as small spheres.
// Each "on" cell becomes one little matte sphere, so the text flows through the
// exact same closest()/trace() path as everything else: it gets lit, casts and
// receives shadows, and shows up in reflections, with no new intersection code.
// ----------------------------------------------------------------------------
// 8x8 monochrome ASCII font (U+0000..U+007F). Public Domain.
// Source: Daniel Hepper, github.com/dhepper/font8x8 (IBM VGA fonts).
// Each glyph = 8 row-bytes; bit 0 (LSB) is the LEFTMOST column.
static const unsigned char font8x8[128][8] = {
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0000 (nul)
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0001
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0002
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0003
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0004
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0005
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0006
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0007
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0008
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0009
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+000A
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+000B
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+000C
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+000D
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+000E
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+000F
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0010
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0011
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0012
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0013
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0014
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0015
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0016
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0017
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0018
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0019
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+001A
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+001B
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+001C
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+001D
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+001E
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+001F
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0020 (space)
  { 0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00},   // U+0021 (!)
  { 0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0022 (")
  { 0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00},   // U+0023 (#)
  { 0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00},   // U+0024 ($)
  { 0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00},   // U+0025 (%)
  { 0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00},   // U+0026 (&)
  { 0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0027 (')
  { 0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00},   // U+0028 (()
  { 0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00},   // U+0029 ())
  { 0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00},   // U+002A (*)
  { 0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00},   // U+002B (+)
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x06},   // U+002C (,)
  { 0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00},   // U+002D (-)
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00},   // U+002E (.)
  { 0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00},   // U+002F (/)
  { 0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00},   // U+0030 (0)
  { 0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00},   // U+0031 (1)
  { 0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00},   // U+0032 (2)
  { 0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00},   // U+0033 (3)
  { 0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00},   // U+0034 (4)
  { 0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00},   // U+0035 (5)
  { 0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00},   // U+0036 (6)
  { 0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00},   // U+0037 (7)
  { 0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00},   // U+0038 (8)
  { 0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00},   // U+0039 (9)
  { 0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00},   // U+003A (:)
  { 0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x06},   // U+003B (;)
  { 0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00},   // U+003C (<)
  { 0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00},   // U+003D (=)
  { 0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00},   // U+003E (>)
  { 0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00},   // U+003F (?)
  { 0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00},   // U+0040 (@)
  { 0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00},   // U+0041 (A)
  { 0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00},   // U+0042 (B)
  { 0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00},   // U+0043 (C)
  { 0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00},   // U+0044 (D)
  { 0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00},   // U+0045 (E)
  { 0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00},   // U+0046 (F)
  { 0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00},   // U+0047 (G)
  { 0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00},   // U+0048 (H)
  { 0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00},   // U+0049 (I)
  { 0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00},   // U+004A (J)
  { 0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00},   // U+004B (K)
  { 0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00},   // U+004C (L)
  { 0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00},   // U+004D (M)
  { 0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00},   // U+004E (N)
  { 0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00},   // U+004F (O)
  { 0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00},   // U+0050 (P)
  { 0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00},   // U+0051 (Q)
  { 0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00},   // U+0052 (R)
  { 0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00},   // U+0053 (S)
  { 0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00},   // U+0054 (T)
  { 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00},   // U+0055 (U)
  { 0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00},   // U+0056 (V)
  { 0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00},   // U+0057 (W)
  { 0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00},   // U+0058 (X)
  { 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00},   // U+0059 (Y)
  { 0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00},   // U+005A (Z)
  { 0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E, 0x00},   // U+005B ([)
  { 0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00},   // U+005C (\)
  { 0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E, 0x00},   // U+005D (])
  { 0x08, 0x1C, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00},   // U+005E (^)
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF},   // U+005F (_)
  { 0x0C, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+0060 (`)
  { 0x00, 0x00, 0x1E, 0x30, 0x3E, 0x33, 0x6E, 0x00},   // U+0061 (a)
  { 0x07, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00},   // U+0062 (b)
  { 0x00, 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00},   // U+0063 (c)
  { 0x38, 0x30, 0x30, 0x3e, 0x33, 0x33, 0x6E, 0x00},   // U+0064 (d)
  { 0x00, 0x00, 0x1E, 0x33, 0x3f, 0x03, 0x1E, 0x00},   // U+0065 (e)
  { 0x1C, 0x36, 0x06, 0x0f, 0x06, 0x06, 0x0F, 0x00},   // U+0066 (f)
  { 0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x1F},   // U+0067 (g)
  { 0x07, 0x06, 0x36, 0x6E, 0x66, 0x66, 0x67, 0x00},   // U+0068 (h)
  { 0x0C, 0x00, 0x0E, 0x0C, 0x0C, 0x0C, 0x1E, 0x00},   // U+0069 (i)
  { 0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E},   // U+006A (j)
  { 0x07, 0x06, 0x66, 0x36, 0x1E, 0x36, 0x67, 0x00},   // U+006B (k)
  { 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00},   // U+006C (l)
  { 0x00, 0x00, 0x33, 0x7F, 0x7F, 0x6B, 0x63, 0x00},   // U+006D (m)
  { 0x00, 0x00, 0x1F, 0x33, 0x33, 0x33, 0x33, 0x00},   // U+006E (n)
  { 0x00, 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00},   // U+006F (o)
  { 0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x0F},   // U+0070 (p)
  { 0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x78},   // U+0071 (q)
  { 0x00, 0x00, 0x3B, 0x6E, 0x66, 0x06, 0x0F, 0x00},   // U+0072 (r)
  { 0x00, 0x00, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x00},   // U+0073 (s)
  { 0x08, 0x0C, 0x3E, 0x0C, 0x0C, 0x2C, 0x18, 0x00},   // U+0074 (t)
  { 0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6E, 0x00},   // U+0075 (u)
  { 0x00, 0x00, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00},   // U+0076 (v)
  { 0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00},   // U+0077 (w)
  { 0x00, 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00},   // U+0078 (x)
  { 0x00, 0x00, 0x33, 0x33, 0x33, 0x3E, 0x30, 0x1F},   // U+0079 (y)
  { 0x00, 0x00, 0x3F, 0x19, 0x0C, 0x26, 0x3F, 0x00},   // U+007A (z)
  { 0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38, 0x00},   // U+007B ({)
  { 0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00},   // U+007C (|)
  { 0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07, 0x00},   // U+007D (})
  { 0x6E, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},   // U+007E (~)
  { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}    // U+007F
};

// Length of one line of text (chars up to the next '\n' or end of string).
static int beadLineLen(const char* s) {
  int n = 0;
  while (*s && *s != '\n') { ++n; ++s; }
  return n;
}

// Render arbitrary ASCII text as beads, centered on `center`.
//   * Any printable ASCII works; non-ASCII bytes (>=128) draw as '?'.
//   * '\n' starts a new line; lines are individually centered and stacked.
//   * Letters lie in the plane z == center.z, facing +z (toward the camera).
//   * Each drawn cell becomes a sphere of radius `bead`; `spacing` is the
//     center-to-center bead distance (use bead ~0.6*spacing for joined strokes).
//   * Colors cycle through letterColors[].
// "Within reason": every lit cell is a sphere and closest() is a linear scan,
// so cost grows with total bead count. A short banner is cheap; a paragraph at
// this font size can add thousands of spheres and slow the render noticeably.
static void addBeadText(std::vector<Sphere>& scene, const char* text, Vec3 center,
                        float spacing, float bead,
                        const Vec3* letterColors, int numColors) {
  const float advance = 9.0f * spacing;  // per cell: 8 columns + 1 blank gap
  const float glyphExt = 7.0f * spacing; // span from first to last bead within a glyph

  int lineCount = 1;
  for (const char* p = text; *p; ++p) if (*p == '\n') ++lineCount;
  float blockH = static_cast<float>(lineCount - 1) * advance + glyphExt;
  float topY   = center.y + blockH * 0.5f;  // y of the top row of beads

  int k = 0;            // color index, advances per character
  int li = 0;           // current line index
  const char* p = text;
  for (;;) {
    int L = beadLineLen(p);
    float lineW  = (L > 0) ? (static_cast<float>(L - 1) * advance + glyphExt) : 0.0f;
    float startX = center.x - lineW * 0.5f;   // x of the leftmost bead on this line
    float baseY  = topY - static_cast<float>(li) * advance;

    int col = 0;
    for (; *p && *p != '\n'; ++p, ++col, ++k) {
      unsigned char ch = static_cast<unsigned char>(*p);
      if (ch >= 128) ch = static_cast<unsigned char>('?');
      const unsigned char* g = font8x8[ch];
      Vec3 c = letterColors[((k % numColors) + numColors) % numColors];
      float xBase = startX + static_cast<float>(col) * advance;
      for (int r = 0; r < 8; ++r) {
        unsigned char rowbits = g[r];
        if (!rowbits) continue;                 // skip empty rows fast
        for (int cc = 0; cc < 8; ++cc) {
          if ((rowbits >> cc) & 1) {            // bit 0 = leftmost column
            Vec3 ctr{ xBase + static_cast<float>(cc) * spacing,
                      baseY - static_cast<float>(r) * spacing,
                      center.z };
            scene.push_back({ ctr, bead, c, 0.0f, false });
          }
        }
      }
    }
    if (*p == '\n') { ++p; ++li; continue; }
    break;
  }
}


// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int main() {
  const int   W = 900, H = 540;
  const int   MAX_DEPTH = 5;
  const float aspect = static_cast<float>(W) / static_cast<float>(H);

  // --- Scene (all hard-coded) ---
  std::vector<Sphere> scene;
  scene.push_back({{0, -1000, 0}, 1000.f, {0, 0, 0},        0.00f, true});  // ground
  scene.push_back({{0,     1,  0},   1.0f, {0.80f,0.30f,0.30f}, 0.10f,false}); // center, glossy red
  scene.push_back({{-2.3f, 1,  0},   1.0f, {0.92f,0.92f,0.92f}, 0.88f,false}); // left, mirror
  scene.push_back({{ 2.3f, 1,  0},   1.0f, {0.30f,0.45f,0.90f}, 0.30f,false}); // right, blue
  scene.push_back({{ 0.0f, 0.45f, 1.6f}, 0.45f, {0.95f,0.78f,0.25f}, 0.0f,false}); // little gold

  // A ring of small spheres for visual interest.
  const Vec3 palette[5] = {
    {0.85f,0.25f,0.45f}, {0.25f,0.75f,0.55f}, {0.85f,0.55f,0.20f},
    {0.55f,0.35f,0.85f}, {0.20f,0.70f,0.85f}
  };
  for (int i = 0; i < 7; ++i) {
    float ang = static_cast<float>(i) / 7.0f * 2.0f * 3.14159265f;
    float rad = 3.4f;
    Vec3  c{ std::cos(ang) * rad, 0.35f, std::sin(ang) * rad - 0.4f };
    float refl = (i % 3 == 0) ? 0.55f : 0.0f;
    scene.push_back({ c, 0.35f, palette[i % 5], refl, false });
  }

  Light  light{ {6.0f, 7.0f, 4.0f}, {1.0f, 0.98f, 0.92f}, 0.95f };
  Camera cam({0.0f, 1.7f, 5.2f}, {0.0f, 0.9f, 0.0f}, {0, 1, 0}, 42.0f, aspect);

  // Beaded text floating in the background, up above the spheres.
  // Per-letter colors so it reads as bright balloon lettering against the sky.
  // Arbitrary ASCII works; use '\n' for multiple lines (each line is centered).
  const Vec3 textColors[6] = {
    {0.95f,0.25f,0.30f}, {0.97f,0.55f,0.15f}, {0.98f,0.82f,0.20f},
    {0.30f,0.78f,0.45f}, {0.25f,0.55f,0.95f}, {0.65f,0.35f,0.90f}
  };
  {
    const char* msg     = "I hecking love \n SSDs!!!";
    const float spacing = 0.10f;        // bead center-to-center distance
    const float bead    = 0.062f;       // bead radius (slight overlap -> joined strokes)
    Vec3 center{ 0.0f, 2.95f, -7.0f };  // block is centered here, well behind the scene
    addBeadText(scene, msg, center, spacing, bead, textColors, 6);
  }

  // 2x2 ordered sub-pixel offsets for deterministic anti-aliasing.
  const float offs[4][2] = {{0.25f,0.25f},{0.75f,0.25f},{0.25f,0.75f},{0.75f,0.75f}};

  std::cerr << "Rendering " << W << "x" << H
            << " on " << parlay::num_workers() << " workers...\n";

  // MAP over pixels.
  parlay::sequence<Vec3> pixels = parlay::tabulate(static_cast<size_t>(W) * H, [&](size_t idx) {
    int px = static_cast<int>(idx % W);
    int py = static_cast<int>(idx / W);
    Vec3 acc(0, 0, 0);
    for (int s = 0; s < 4; ++s) {
      Ray r = cam.ray((px + offs[s][0]) / W, (py + offs[s][1]) / H);
      acc = acc + trace(r, scene, light, MAX_DEPTH);
    }
    acc = acc * 0.25f;
    auto g = [](float v) { return std::sqrt(std::min(1.0f, std::max(0.0f, v))); }; // gamma 2
    return Vec3(g(acc.x), g(acc.y), g(acc.z));
  });

  // REDUCE: mean luminance, just to show an aggregate (printed to stderr).
  double meanLum = parlay::reduce(parlay::map(pixels, [](Vec3 c) {
                     return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
                   })) / static_cast<double>(pixels.size());
  std::cerr << "Mean luminance: " << meanLum << "\n";

  // MAP colors -> bytes.
  parlay::sequence<unsigned char> bytes =
      parlay::tabulate(static_cast<size_t>(W) * H * 3, [&](size_t i) {
        Vec3 c = pixels[i / 3];
        float v = c[static_cast<int>(i % 3)];
        return static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, v * 255.0f + 0.5f)));
      });

  // Write binary P6 PPM to stdout (only image data on stdout).
  std::cout << "P6\n" << W << " " << H << "\n255\n";
  std::cout.write(reinterpret_cast<const char*>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
  std::cout.flush();
  return 0;
}
