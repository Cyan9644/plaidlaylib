// path_tracer.cpp
//
// Iterative path tracer: each bounce level's ray state lives on the SSDs as a
// chunk_seq<RayState>.  One ChunkMap pass advances all rays by one bounce;
// ChunkFilter then removes terminated rays so subsequent passes only touch
// active ones.  Previous-bounce SSD files are deleted immediately to bound
// disk usage.
//
// Supports OBJ meshes (MTL + PNG textures via stb_image) alongside the sphere
// scene.  Triangle intersection is accelerated by a median-split BVH.
//
// Build:
//   make bazel-bin/pathTracer
// Run:
//   bazel-bin/pathTracer
//   python3 ChunkSequence/examples/make_png.py ipt_meta.txt path_traced.png
//
// OBJ mesh: place mario.obj (+ .mtl + textures) in the working directory.
// Tune OBJ_SCALE so the mesh is ~2 world units tall; OBJ_OFFSET positions it.

#include <parlay/primitives.h>
#include <parlay/sequence.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "ChunkSequence/chunk_filter.h"
#include "ChunkSequence/chunk_map.h"
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_seq_reader.h"
#include "utils/command_line.h"
#include "utils/file_utils.h"

// ─────────────────────────────────────────────────────────────────────────────
// Vec2 / Vec3
// ─────────────────────────────────────────────────────────────────────────────
struct Vec2 { float u = 0, v = 0; };

struct Vec3 {
  float x = 0, y = 0, z = 0;
  Vec3() = default;
  Vec3(float a, float b, float c) : x(a), y(b), z(c) {}
  float operator[](int i) const { return i==0 ? x : (i==1 ? y : z); }
};

static inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
static inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
static inline Vec3 operator-(Vec3 a)          { return {-a.x,-a.y,-a.z}; }
static inline Vec3 operator*(Vec3 a, float s) { return {a.x*s,a.y*s,a.z*s}; }
static inline Vec3 operator*(float s, Vec3 a) { return a*s; }
static inline Vec3 operator*(Vec3 a, Vec3 b)  { return {a.x*b.x,a.y*b.y,a.z*b.z}; }
static inline Vec3 operator/(Vec3 a, float s) { return {a.x/s,a.y/s,a.z/s}; }

static inline float dot(Vec3 a, Vec3 b)     { return a.x*b.x+a.y*b.y+a.z*b.z; }
static inline float length(Vec3 a)          { return std::sqrt(dot(a,a)); }
static inline Vec3  normalize(Vec3 a)       { return a/length(a); }
static inline Vec3  cross(Vec3 a, Vec3 b)  {
  return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
static inline Vec3 reflect(Vec3 v, Vec3 n)  { return v - n*(2.f*dot(v,n)); }
static inline Vec3 vmin(Vec3 a, Vec3 b) {
  return {std::min(a.x,b.x),std::min(a.y,b.y),std::min(a.z,b.z)};
}
static inline Vec3 vmax(Vec3 a, Vec3 b) {
  return {std::max(a.x,b.x),std::max(a.y,b.y),std::max(a.z,b.z)};
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene types
// ─────────────────────────────────────────────────────────────────────────────
struct Ray { Vec3 orig, dir; Vec3 at(float t) const { return orig+dir*t; } };

struct Sphere {
  Vec3  center; float radius;
  Vec3  albedo; float reflectivity;
  bool  checker;
};

struct Material {
  Vec3  albedo       = {0.8f,0.8f,0.8f};
  float reflectivity = 0.f;
  int   tex_id       = -1;
};

struct Texture {
  int w=0, h=0;
  std::vector<uint8_t> data; // packed RGB

  bool load(const std::string& path) {
    int ch;
    uint8_t* raw = stbi_load(path.c_str(), &w, &h, &ch, 3);
    if (!raw) { std::cerr << "  [tex] failed: " << path << "\n"; return false; }
    data.assign(raw, raw + w*h*3);
    stbi_image_free(raw);
    std::cerr << "  [tex] " << path << " (" << w << "x" << h << ")\n";
    return true;
  }

  Vec3 sample(float u, float v) const {
    if (data.empty()) return {0.8f,0.8f,0.8f};
    u -= std::floor(u);
    v  = 1.f - (v - std::floor(v)); // OBJ V=0 at bottom; image row 0 at top
    float px = u*(w-1), py = v*(h-1);
    int x0=(int)px, y0=(int)py;
    int x1=std::min(x0+1,w-1), y1=std::min(y0+1,h-1);
    float fx=px-x0, fy=py-y0;
    auto p = [&](int x,int y) -> Vec3 {
      const uint8_t* d = data.data()+(y*w+x)*3;
      return {d[0]/255.f, d[1]/255.f, d[2]/255.f};
    };
    return p(x0,y0)*(1-fx)*(1-fy) + p(x1,y0)*fx*(1-fy)
         + p(x0,y1)*(1-fx)*fy    + p(x1,y1)*fx*fy;
  }
};

struct Triangle {
  Vec3 v0,v1,v2;
  Vec3 n0,n1,n2;
  Vec2 uv0,uv1,uv2;
  int  mat_idx=0;
};

struct Hit {
  bool hit=false; float t=0;
  Vec3 point, normal;
  int  idx=-1;       // sphere index, or material index for triangles
  bool is_tri=false;
  Vec2 uv;
};

struct Light { Vec3 pos; Vec3 color; float intensity; };

// ─────────────────────────────────────────────────────────────────────────────
// AABB
// ─────────────────────────────────────────────────────────────────────────────
struct AABB {
  Vec3 mn{1e30f,1e30f,1e30f}, mx{-1e30f,-1e30f,-1e30f};
  void expand(Vec3 p)       { mn=vmin(mn,p); mx=vmax(mx,p); }
  void expand(const AABB& b){ mn=vmin(mn,b.mn); mx=vmax(mx,b.mx); }
  Vec3 centroid() const     { return (mn+mx)*0.5f; }
  bool hit(const Ray& r, float tmin, float tmax) const {
    for (int a=0; a<3; ++a) {
      float inv=1.f/r.dir[a];
      float t0=(mn[a]-r.orig[a])*inv, t1=(mx[a]-r.orig[a])*inv;
      if (inv<0.f) std::swap(t0,t1);
      tmin=std::max(tmin,t0); tmax=std::min(tmax,t1);
      if (tmax<=tmin) return false;
    }
    return true;
  }
};

static AABB tri_aabb(const Triangle& t) {
  AABB b; b.expand(t.v0); b.expand(t.v1); b.expand(t.v2); return b;
}

// ─────────────────────────────────────────────────────────────────────────────
// BVH  (median-split, flat node array)
// ─────────────────────────────────────────────────────────────────────────────
struct BVH {
  struct Node {
    AABB box;
    int  left=-1, right=-1;
    int  tri_start=0, tri_count=0;
    bool is_leaf() const { return tri_count>0; }
  };
  std::vector<Node> nodes;
  std::vector<int>  order;
  const std::vector<Triangle>* tris = nullptr;

  void build(const std::vector<Triangle>& t) {
    if (t.empty()) return;
    tris = &t;
    order.resize(t.size());
    std::iota(order.begin(), order.end(), 0);
    nodes.reserve(t.size()*2);
    buildNode(0, (int)t.size());
  }

  Hit traverse(const Ray& r, float tmin, float tmax) const {
    Hit best;
    if (nodes.empty()) return best;
    traverseNode(0, r, tmin, tmax, best);
    return best;
  }

private:
  int buildNode(int lo, int hi) {
    int idx = (int)nodes.size();
    nodes.push_back({});

    AABB box;
    for (int i=lo; i<hi; ++i) box.expand(tri_aabb((*tris)[order[i]]));
    nodes[idx].box = box;

    if (hi-lo <= 4) {
      nodes[idx].tri_start = lo;
      nodes[idx].tri_count = hi-lo;
      return idx;
    }

    Vec3 ext = box.mx - box.mn;
    int axis = (ext.x>ext.y && ext.x>ext.z) ? 0 : (ext.y>ext.z ? 1 : 2);
    int mid  = (lo+hi)/2;
    std::nth_element(order.begin()+lo, order.begin()+mid, order.begin()+hi,
      [&](int a, int b){
        return tri_aabb((*tris)[a]).centroid()[axis]
             < tri_aabb((*tris)[b]).centroid()[axis];
      });

    int L = buildNode(lo,  mid);
    int R = buildNode(mid, hi);
    // re-index after recursive pushes (vector may have reallocated)
    nodes[idx].left  = L;
    nodes[idx].right = R;
    return idx;
  }

  void traverseNode(int ni, const Ray& r, float tmin, float tmax, Hit& best) const {
    const Node& nd = nodes[ni];
    if (!nd.box.hit(r, tmin, tmax)) return;
    if (nd.is_leaf()) {
      for (int i=nd.tri_start; i<nd.tri_start+nd.tri_count; ++i) {
        const Triangle& tri = (*tris)[order[i]];
        Vec3 e1=tri.v1-tri.v0, e2=tri.v2-tri.v0;
        Vec3 pv=cross(r.dir, e2);
        float det=dot(e1,pv);
        if (std::abs(det)<1e-8f) continue;
        float inv=1.f/det;
        Vec3  tv=r.orig-tri.v0;
        float u=dot(tv,pv)*inv;
        if (u<0.f||u>1.f) continue;
        Vec3  qv=cross(tv,e1);
        float v=dot(r.dir,qv)*inv;
        if (v<0.f||u+v>1.f) continue;
        float t=dot(e2,qv)*inv;
        if (t<tmin||t>=tmax) continue;
        float w=1.f-u-v;
        best.hit    = true;
        best.t      = t;
        best.point  = r.at(t);
        best.normal = normalize(tri.n0*w + tri.n1*u + tri.n2*v);
        best.uv     = {tri.uv0.u*w+tri.uv1.u*u+tri.uv2.u*v,
                       tri.uv0.v*w+tri.uv1.v*u+tri.uv2.v*v};
        best.idx    = tri.mat_idx;
        best.is_tri = true;
        tmax = t; // tighten window
      }
    } else {
      traverseNode(nd.left,  r, tmin, tmax, best);
      if (best.hit) tmax = best.t;
      traverseNode(nd.right, r, tmin, tmax, best);
    }
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Shading helpers
// ─────────────────────────────────────────────────────────────────────────────
static inline Vec3 skyColor(Vec3 dir) {
  float t=0.5f*(normalize(dir).y+1.f);
  return Vec3(1,1,1)*(1.f-t) + Vec3(0.5f,0.7f,1.f)*t;
}

static inline Vec3 checkerColor(Vec3 p) {
  int c=static_cast<int>(std::floor(p.x*0.6f)+std::floor(p.z*0.6f));
  return ((c&1)==0) ? Vec3(0.85f,0.85f,0.85f) : Vec3(0.12f,0.22f,0.32f);
}

static Hit closest(const Ray& r,
                   const std::vector<Sphere>& spheres,
                   const BVH& bvh,
                   float tmin, float tmax) {
  // Test spheres
  int   best_s  = -1;
  float best_st = tmax;
  for (int i=0; i<(int)spheres.size(); ++i) {
    const Sphere& s=spheres[i];
    Vec3  oc=r.orig-s.center;
    float a=dot(r.dir,r.dir), hb=dot(oc,r.dir);
    float c=dot(oc,oc)-s.radius*s.radius;
    float disc=hb*hb-a*c;
    if (disc<0.f) continue;
    float sq=std::sqrt(disc);
    float t=(-hb-sq)/a;
    if (t<tmin||t>best_st){ t=(-hb+sq)/a; if(t<tmin||t>best_st) continue; }
    best_s=i; best_st=t;
  }

  // BVH (use sphere tmax to cull)
  Hit tri_hit = bvh.traverse(r, tmin, best_s>=0 ? best_st : tmax);

  if (tri_hit.hit && (best_s<0 || tri_hit.t<best_st))
    return tri_hit;

  if (best_s>=0) {
    Hit h;
    h.hit    = true;
    h.t      = best_st;
    h.is_tri = false;
    h.idx    = best_s;
    h.point  = r.at(best_st);
    h.normal = normalize(h.point - spheres[best_s].center);
    return h;
  }
  return {};
}

// ─────────────────────────────────────────────────────────────────────────────
// OBJ / MTL loader
// ─────────────────────────────────────────────────────────────────────────────
static std::string dirOf(const std::string& p) {
  auto pos=p.rfind('/'); return pos==std::string::npos ? "." : p.substr(0,pos);
}

// Parse one OBJ vertex spec: "v", "v/t", "v//n", "v/t/n"  (1-based or negative)
struct VI { int v=-1,t=-1,n=-1; };
static VI parseVI(const std::string& s, int nv, int nt, int nn) {
  auto fix=[](int i,int n)->int{ return i>0?i-1:(i<0?n+i:-1); };
  VI vi;
  size_t p1=s.find('/');
  if (p1==std::string::npos) { vi.v=std::stoi(s); }
  else {
    vi.v=std::stoi(s.substr(0,p1));
    size_t p2=s.find('/',p1+1);
    if (p2==std::string::npos) {
      if (p1+1<s.size()) vi.t=std::stoi(s.substr(p1+1));
    } else {
      if (p2>p1+1) vi.t=std::stoi(s.substr(p1+1,p2-p1-1));
      if (p2+1<s.size()) vi.n=std::stoi(s.substr(p2+1));
    }
  }
  vi.v=fix(vi.v,nv); vi.t=fix(vi.t,nt); vi.n=fix(vi.n,nn);
  return vi;
}

static void loadMTL(const std::string& path, const std::string& base,
                    std::map<std::string,int>& mat_map,
                    std::vector<Material>& mats,
                    std::vector<Texture>& textures) {
  std::ifstream f(path);
  if (!f) { std::cerr << "  [mtl] not found: " << path << "\n"; return; }
  std::string line, cur;
  while (std::getline(f,line)) {
    if (!line.empty() && line.back()=='\r') line.pop_back();
    std::istringstream ss(line); std::string tok; ss>>tok;
    if (tok=="newmtl") {
      ss>>cur;
      if (!mat_map.count(cur)) { mat_map[cur]=(int)mats.size(); mats.push_back({}); }
    } else if (!cur.empty()) {
      int mi=mat_map[cur];
      if (tok=="Kd") {
        ss>>mats[mi].albedo.x>>mats[mi].albedo.y>>mats[mi].albedo.z;
      } else if (tok=="Ns") {
        float ns; ss>>ns; mats[mi].reflectivity=std::min(1.f,ns/500.f)*0.25f;
      } else if (tok=="map_Kd") {
        std::string tp; ss>>tp;
        if (!tp.empty()&&tp[0]!='/') tp=base+"/"+tp;
        int ti=(int)textures.size(); textures.push_back({});
        if (textures.back().load(tp)) mats[mi].tex_id=ti;
        else textures.pop_back();
      }
    }
  }
}

// Load an OBJ file. scale is applied to positions, then offset is added.
static bool loadOBJ(const std::string& path, float scale, Vec3 offset,
                    std::function<Vec3(Vec3)> axis_remap,
                    std::vector<Triangle>& out_tris,
                    std::vector<Material>& out_mats,
                    std::vector<Texture>& out_textures) {
  std::ifstream f(path);
  if (!f) { std::cerr << "[obj] not found: " << path << "\n"; return false; }
  std::cerr << "[obj] loading " << path << "\n";

  std::string base=dirOf(path);
  std::map<std::string,int> mat_map;
  if (out_mats.empty()) out_mats.push_back({}); // default material index 0

  std::vector<Vec3> pos, nrm;
  std::vector<Vec2> tex;
  int cur_mat=0;

  std::string line;
  while (std::getline(f,line)) {
    if (!line.empty()&&line.back()=='\r') line.pop_back();
    if (line.empty()||line[0]=='#') continue;
    std::istringstream ss(line); std::string tok; ss>>tok;

    if (tok=="v") {
      Vec3 p; ss>>p.x>>p.y>>p.z; pos.push_back(axis_remap(p)*scale+offset);
    } else if (tok=="vt") {
      Vec2 t; ss>>t.u>>t.v; tex.push_back(t);
    } else if (tok=="vn") {
      Vec3 n; ss>>n.x>>n.y>>n.z; nrm.push_back(normalize(axis_remap(n)));
    } else if (tok=="mtllib") {
      std::string mf; ss>>mf;
      if (!mf.empty()&&mf[0]!='/') mf=base+"/"+mf;
      loadMTL(mf, base, mat_map, out_mats, out_textures);
    } else if (tok=="usemtl") {
      std::string nm; ss>>nm;
      auto it=mat_map.find(nm);
      cur_mat=(it!=mat_map.end())?it->second:0;
    } else if (tok=="f") {
      std::vector<VI> verts;
      std::string grp;
      while (ss>>grp)
        verts.push_back(parseVI(grp,(int)pos.size(),(int)tex.size(),(int)nrm.size()));

      // Fan-triangulate
      for (int i=1; i+1<(int)verts.size(); ++i) {
        auto gp=[&](const VI& vi)->Vec3 { return pos[vi.v]; };
        auto gn=[&](const VI& vi)->Vec3 {
          if (vi.n>=0) return normalize(nrm[vi.n]);
          return normalize(cross(pos[verts[1].v]-pos[verts[0].v],
                                 pos[verts[2].v]-pos[verts[0].v]));
        };
        auto gt=[&](const VI& vi)->Vec2 {
          return vi.t>=0 ? tex[vi.t] : Vec2{};
        };
        Triangle tri;
        tri.v0=gp(verts[0]); tri.v1=gp(verts[i]); tri.v2=gp(verts[i+1]);
        tri.n0=gn(verts[0]); tri.n1=gn(verts[i]); tri.n2=gn(verts[i+1]);
        tri.uv0=gt(verts[0]);tri.uv1=gt(verts[i]);tri.uv2=gt(verts[i+1]);
        tri.mat_idx=cur_mat;
        out_tris.push_back(tri);
      }
    }
  }
  std::cerr << "[obj] " << out_tris.size() << " triangles, "
            << out_mats.size() << " materials, "
            << out_textures.size() << " textures\n";
  return !out_tris.empty();
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera
// ─────────────────────────────────────────────────────────────────────────────
struct Camera {
  Vec3  pos, forward, right, up;
  float halfW, halfH;
  Camera(Vec3 from, Vec3 to, Vec3 wu, float vfov, float aspect) {
    pos     = from;
    forward = normalize(to-from);
    right   = normalize(cross(forward,wu));
    up      = cross(right,forward);
    halfH   = std::tan(vfov*0.5f*3.14159265358979f/180.f);
    halfW   = aspect*halfH;
  }
  Ray ray(float sx, float sy) const {
    float u=(2.f*sx-1.f)*halfW, v=(1.f-2.f*sy)*halfH;
    return {pos, normalize(forward+right*u+up*v)};
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// Beaded text  (verbatim — used for the sphere scene label)
// ─────────────────────────────────────────────────────────────────────────────
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

static int beadLineLen(const char* s) {
  int n=0; while(*s&&*s!='\n'){++n;++s;} return n;
}

static void addBeadText(std::vector<Sphere>& sc, const char* text, Vec3 center,
                        float spacing, float bead,
                        const Vec3* colors, int nc) {
  const float adv=9.f*spacing, ext=7.f*spacing;
  int lc=1; for(const char* p=text;*p;++p) if(*p=='\n') ++lc;
  float bh=(float)(lc-1)*adv+ext, topY=center.y+bh*0.5f;
  int k=0,li=0; const char* p=text;
  for(;;){
    int L=beadLineLen(p);
    float lw=L>0?(float)(L-1)*adv+ext:0.f, sx=center.x-lw*0.5f;
    float by=topY-(float)li*adv;
    int col=0;
    for(;*p&&*p!='\n';++p,++col,++k){
      unsigned char ch=(unsigned char)*p; if(ch>=128) ch='?';
      const unsigned char* g=font8x8[ch];
      Vec3 c=colors[((k%nc)+nc)%nc];
      float xb=sx+(float)col*adv;
      for(int r=0;r<8;++r){ unsigned char row=g[r]; if(!row) continue;
        for(int cc=0;cc<8;++cc) if((row>>cc)&1)
          sc.push_back({{xb+(float)cc*spacing, by-(float)r*spacing, center.z},
                        bead, c, 0.f, false});
      }
    }
    if(*p=='\n'){++p;++li;continue;} break;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ray state carried between bounce passes on SSD
// ─────────────────────────────────────────────────────────────────────────────
struct RayState {
  float ox,oy,oz;     // ray origin
  float dx,dy,dz;     // ray direction
  float tx,ty,tz;     // throughput
  float lx,ly,lz;     // accumulated radiance
  uint32_t pixel_idx; // output pixel
  uint32_t depth;     // bounce count (used in RNG seed)
  uint32_t _pad[2];   // _pad[1] = sample index for RNG decorrelation
};
static_assert(sizeof(RayState)==64,
  "RayState must be 64 bytes so CHUNK_SIZE % sizeof(RayState) == 0");

// 16 bytes → CHUNK_SIZE/16 = 262144 elements/chunk, exactly 4 MB (O_DIRECT safe)
struct AccumPixel { float r, g, b, _pad; };
static_assert(sizeof(AccumPixel)==16, "");
static constexpr size_t ACCUM_EPK = CHUNK_SIZE / sizeof(AccumPixel);

static uint32_t hash32(uint32_t x) {
  x^=x<<13; x^=x>>17; x^=x<<5; return x;
}

static Vec3 cosineHemisphere(Vec3 N, uint32_t seed) {
  uint32_t h=hash32(seed);
  float u1=(h&0xFFFFu)*(1.f/65536.f); h=hash32(h);
  float u2=(h&0xFFFFu)*(1.f/65536.f);
  float r=std::sqrt(u1), phi=2.f*3.14159265358979f*u2;
  float lx=r*std::cos(phi), ly=r*std::sin(phi);
  float lz=std::sqrt(std::max(0.f,1.f-u1));
  Vec3 up=std::abs(N.z)<0.999f?Vec3(0,0,1):Vec3(1,0,0);
  Vec3 T=normalize(cross(up,N)), B=cross(N,T);
  return normalize(T*lx+B*ly+N*lz);
}

static void cleanup_chunk_seq(const chunk_seq& seq) {
  std::set<std::string> seen;
  for (const auto& c : seq.chunks)
    if (seen.insert(c.filename).second) unlink(c.filename.c_str());
}

// Stream a chunk_seq<RayState> and add each ray's radiance into accum[].
// Only rays satisfying pred (or all rays if pred is nullptr) are accumulated.
// Stream rays_seq, group radiance contributions by accum chunk, then for each
// non-empty group: pread the accum chunk, apply updates, pwrite it back.
static void scatter_accum(const chunk_seq& rays_seq, const chunk_seq& accum_seq,
                          bool terminated_only) {
  struct Update { uint32_t local_off; float r, g, b; };
  std::vector<std::vector<Update>> by_chunk(accum_seq.chunks.size());

  ChunkSequenceReader<RayState> reader;
  reader.PrepChunks(rays_seq);
  reader.Start(5, 32, 16);
  while (true) {
    auto [ptr, n, _] = reader.Poll();
    if (!ptr) break;
    for (size_t j=0; j<n; ++j) {
      const RayState& rs=ptr[j];
      if (terminated_only && rs.tx+rs.ty+rs.tz >= 1e-6f) continue;
      size_t pi=(size_t)rs.pixel_idx;
      size_t ci=pi/ACCUM_EPK;
      by_chunk[ci].push_back({(uint32_t)(pi%ACCUM_EPK), rs.lx, rs.ly, rs.lz});
    }
    reader.allocator.Free(ptr);
  }

  std::vector<AccumPixel> buf(ACCUM_EPK);
  for (size_t ci=0; ci<by_chunk.size(); ++ci) {
    if (by_chunk[ci].empty()) continue;
    const chunk& ch=accum_seq.chunks[ci];
    int fd=open(ch.filename.c_str(), O_RDWR);
    pread(fd, buf.data(), CHUNK_SIZE, (off_t)ch.begin_addr);
    for (auto& u : by_chunk[ci]) {
      buf[u.local_off].r+=u.r;
      buf[u.local_off].g+=u.g;
      buf[u.local_off].b+=u.b;
    }
    pwrite(fd, buf.data(), CHUNK_SIZE, (off_t)ch.begin_addr);
    close(fd);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
  ParseGlobalArguments(argc, argv);

  const int   W         = 1920;
  const int   H         = 1080;
  const int   MAX_DEPTH = 5;
  const int   SAMPLES   = 16;
  const float aspect    = (float)W / (float)H;

  // ── OBJ mesh settings ─────────────────────────────────────────────────────
  // Raw bbox: x[-2.02,2.11]  y[12.47,15.09]  z[-3.78,1.21]
  // The exporter used Z-up: character height is along Z (~5 units), depth along Y.
  // axis_remap: (x,y,z) -> (x,z,y)  ← stands him up, chest faces +Z (camera)
  // After remap: height axis = new Y = old Z (range -3.78..1.21)
  //              depth axis  = new Z = old Y (range 12.47..15.09)
  // scale=0.4 -> ~2 units tall.
  // offset.y = +1.513 puts feet at y=0; offset.z = -5.511 centers depth at z=0.
  const char*  OBJ_PATH   = "ChunkSequence/examples/mario/mario.obj";
  const float  OBJ_SCALE  = 0.4f;
  const Vec3   OBJ_OFFSET = {0.f, 1.513f, 5.511f};

  // ── Scene ──────────────────────────────────────────────────────────────────
  std::vector<Sphere> scene;
  scene.push_back({{0,-1000,0},1000.f,{0,0,0},         0.00f,true});
  // Center and front spheres removed — Mario occupies that space.
  scene.push_back({{-2.3f,1,0},   1.f,{0.92f,0.92f,0.92f},0.88f,false});
  scene.push_back({{ 2.3f,1,0},   1.f,{0.30f,0.45f,0.90f},0.30f,false});

  const Vec3 pal[5]={{0.85f,0.25f,0.45f},{0.25f,0.75f,0.55f},{0.85f,0.55f,0.20f},
                     {0.55f,0.35f,0.85f},{0.20f,0.70f,0.85f}};
  for (int i=0; i<7; ++i) {
    float ang=(float)i/7.f*2.f*3.14159265f, rad=3.4f;
    Vec3 c{std::cos(ang)*rad, 0.35f, std::sin(ang)*rad-0.4f};
    scene.push_back({c,0.35f,pal[i%5],(i%3==0)?0.55f:0.f,false});
  }

  Light  light{{6.f,7.f,4.f},{1.f,0.98f,0.92f},0.95f};
  Camera cam({0.f,1.7f,5.2f},{0.f,0.9f,0.f},{0,1,0},42.f,aspect);

  const Vec3 tc[6]={{0.95f,0.25f,0.30f},{0.97f,0.55f,0.15f},{0.98f,0.82f,0.20f},
                    {0.30f,0.78f,0.45f},{0.25f,0.55f,0.95f},{0.65f,0.35f,0.90f}};
  addBeadText(scene,"Get better soon \n Laxman!!!",{0.f,2.95f,-7.f},0.10f,0.062f,tc,6);

  // ── OBJ + BVH ─────────────────────────────────────────────────────────────
  std::vector<Triangle> tris;
  std::vector<Material> mats;
  std::vector<Texture>  textures;
  // Z-up -> Y-up: height to Y, negate old-Y so chest faces +Z (toward camera).
  auto zup_remap = [](Vec3 p) -> Vec3 { return {p.x, p.z, -p.y}; };
  loadOBJ(OBJ_PATH, OBJ_SCALE, OBJ_OFFSET, zup_remap, tris, mats, textures);

  BVH bvh;
  if (!tris.empty()) {
    std::cerr << "[bvh] building over " << tris.size() << " triangles...\n";
    bvh.build(tris);
    std::cerr << "[bvh] " << bvh.nodes.size() << " nodes\n";
  }

  std::cerr << "Rendering " << W << "x" << H
            << "  spp=" << SAMPLES << "  depth=" << MAX_DEPTH
            << "  spheres=" << scene.size()
            << "  triangles=" << tris.size() << "\n";

  // ── Accumulation buffer on SSD (one AccumPixel per pixel) ──────────────────
  chunk_seq accum_seq = ChunkSequenceOps::tabulate<AccumPixel>(
    (size_t)W*H, "ipt_accum",
    std::function<AccumPixel(size_t)>([](size_t){ return AccumPixel{}; }));

  for (int s=0; s<SAMPLES; ++s) {
    std::string suf="_s"+std::to_string(s);

    // ── Phase 1: primary rays ────────────────────────────────────────────────
    chunk_seq rays = ChunkSequenceOps::tabulate<RayState>(
      (size_t)W*H, "ipt_b0"+suf,
      std::function<RayState(size_t)>([&,s](size_t idx)->RayState {
        int px=(int)(idx%W), py=(int)(idx/W);
        uint32_t seed=hash32((uint32_t)idx^((uint32_t)s*2531011u));
        uint32_t h1=hash32(seed), h2=hash32(h1);
        float jx=(h1&0xFFFFu)*(1.f/65536.f)-0.5f;
        float jy=(h2&0xFFFFu)*(1.f/65536.f)-0.5f;
        Ray r=cam.ray((px+0.5f+jx)/W,(py+0.5f+jy)/H);
        RayState rs{};
        rs.ox=r.orig.x; rs.oy=r.orig.y; rs.oz=r.orig.z;
        rs.dx=r.dir.x;  rs.dy=r.dir.y;  rs.dz=r.dir.z;
        rs.tx=rs.ty=rs.tz=1.f;
        rs.lx=rs.ly=rs.lz=0.f;
        rs.pixel_idx=(uint32_t)idx;
        rs.depth=0; rs._pad[0]=0; rs._pad[1]=(uint32_t)s;
        return rs;
      }));
    std::cerr << "  s=" << s << "  b0: " << rays.chunks.size() << " chunks\n";

    // ── Phase 2: bounce loop ─────────────────────────────────────────────────
    for (int d=0; d<MAX_DEPTH; ++d) {
      // Step 1: advance rays one bounce
      chunk_seq bounce = ChunkSequenceOps::ChunkMap<RayState,RayState>(
        rays, "ipt_b"+std::to_string(d+1)+suf,
        std::function<RayState(RayState)>([&](RayState rs)->RayState {
          constexpr float EPS=5e-3f;
          Ray r{{rs.ox,rs.oy,rs.oz},{rs.dx,rs.dy,rs.dz}};
          Hit h=closest(r, scene, bvh, EPS, 1e30f);

          if (!h.hit) {
            Vec3 sky=skyColor(r.dir);
            rs.lx+=rs.tx*sky.x; rs.ly+=rs.ty*sky.y; rs.lz+=rs.tz*sky.z;
            rs.tx=rs.ty=rs.tz=0.f;
            return rs;
          }

          // Albedo and reflectivity depend on surface type
          Vec3  albedo;
          float refl;
          if (!h.is_tri) {
            const Sphere& sp=scene[h.idx];
            albedo = sp.checker ? checkerColor(h.point) : sp.albedo;
            refl   = sp.reflectivity;
          } else {
            const Material& mat=mats[h.idx];
            albedo = mat.tex_id>=0
                   ? textures[mat.tex_id].sample(h.uv.u, h.uv.v)
                   : mat.albedo;
            refl   = mat.reflectivity;
          }

          Vec3 N=h.normal;
          if (dot(N,r.dir)>0.f) N=-N; // face toward incoming ray

          // Direct illumination
          Vec3  toL=light.pos-h.point;
          float dist=length(toL);
          Vec3  L=toL/dist;
          Ray   shad{h.point+N*EPS, L};
          Hit   occ=closest(shad, scene, bvh, EPS, dist-EPS);
          if (!occ.hit) {
            float diff=std::max(0.f,dot(N,L));
            rs.lx+=rs.tx*albedo.x*light.color.x*diff*light.intensity;
            rs.ly+=rs.ty*albedo.y*light.color.y*diff*light.intensity;
            rs.lz+=rs.tz*albedo.z*light.color.z*diff*light.intensity;
          }

          // Next bounce direction
          Vec3 nd;
          if (refl>0.5f) {
            nd=normalize(reflect(r.dir,N));
          } else {
            uint32_t seed=rs.pixel_idx*1664525u
                        ^(rs.depth*22695477u+1u)
                        ^(rs._pad[1]*2531011u);
            nd=cosineHemisphere(N,seed);
          }

          rs.tx*=albedo.x; rs.ty*=albedo.y; rs.tz*=albedo.z;
          rs.ox=h.point.x+N.x*EPS;
          rs.oy=h.point.y+N.y*EPS;
          rs.oz=h.point.z+N.z*EPS;
          rs.dx=nd.x; rs.dy=nd.y; rs.dz=nd.z;
          rs.depth++;
          return rs;
        }));
      cleanup_chunk_seq(rays);

      // Step 2: collect terminated rays into accum
      scatter_accum(bounce, accum_seq, /*terminated_only=*/true);

      // Step 3: filter to active rays only
      rays = ChunkSequenceOps::ChunkFilter<RayState>(
        bounce, "ipt_f"+std::to_string(d)+suf,
        std::function<bool(RayState)>([](RayState rs)->bool {
          return rs.tx+rs.ty+rs.tz >= 1e-6f;
        }));
      cleanup_chunk_seq(bounce);

      std::cerr << "    b" << (d+1) << ": " << rays.chunks.size() << " chunks\n";
      if (rays.chunks.empty()) break;
    }

    // ── Phase 3: accumulate depth-exhausted active rays ──────────────────────
    scatter_accum(rays, accum_seq, /*terminated_only=*/false);
    cleanup_chunk_seq(rays);
    std::cerr << "  sample " << (s+1) << "/" << SAMPLES << " done\n";
  }

  // ── Phase 4: tonemap accum_seq → uint32_t pixels via ChunkMap ───────────────
  const float inv=1.f/(float)SAMPLES;
  chunk_seq pixels = ChunkSequenceOps::ChunkMap<AccumPixel,uint32_t>(
    accum_seq, "ipt_pixels",
    std::function<uint32_t(AccumPixel)>([inv](AccumPixel p)->uint32_t {
      auto g=[](float v){ return std::sqrt(std::min(1.f,std::max(0.f,v))); };
      auto b=[&](float v){ return (uint32_t)(g(v)*255.f+0.5f); };
      return (b(p.r*inv)<<16)|(b(p.g*inv)<<8)|b(p.b*inv);
    }));
  cleanup_chunk_seq(accum_seq);

  std::cerr << "Done. " << pixels.chunks.size() << " output chunks on "
            << GetSSDList().size() << " drive(s).\n";

  {
    std::ofstream meta("ipt_meta.txt");
    meta << W << " " << H << "\n";
    for (const auto& c : pixels.chunks)
      meta << c.index << " " << c.filename << " " << c.begin_addr << " " << c.used << "\n";
  }
  std::cerr << "Run: python3 ChunkSequence/examples/make_png.py ipt_meta.txt path_traced.png\n";
  return 0;
}
