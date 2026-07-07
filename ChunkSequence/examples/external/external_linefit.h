#ifndef EXTERNAL_LINEFIT_H
#define EXTERNAL_LINEFIT_H
#include <utility>

#include <parlay/primitives.h>
#include <parlay/delayed.h>
#include "ChunkSequence/chunk_seq.h"
#include "ChunkSequence/chunk_delayed.h"

namespace ChunkSequenceOps {

using point = std::pair<double,double>;

auto f = [] (point a, point b) {
  return point{a.first + b.first, a.second + b.second};};
auto add_points = parlay::binary_op(f, point(0,0));

//we assume that x and y are the same length and type of chunk_seq
//okay, delaying is not that obvious here so I'll write some comments to explain
auto linefit(const chunk_seq& x, const chunk_seq& y) {

    //this function is intended to fit a set of points to a line
size_t n = 0;
  for(size_t r = 0; r < x.chunks.size(); r++){
n+= x.chunks[r].used;
  }
  n/=sizeof(double);
//find number of elements (point pairs). This should apply for both sequences because they're of the same underlying type and each x must have a y

  //zip the two external sequences into a single external sequence, accessed delayed so that we don't instantiate the entire sequence on disk 
  //I don't really know what casting to delayed::delay doeos here
  auto zipper = ChunkSequenceOps::delayed::zip(ChunkSequenceOps::delayed::delay<double>(x), ChunkSequenceOps::delayed::delay<double>(y));

  //without instantiating the sequence, access its elements individually and compute the sums for both axis points
  auto [xsum, ysum] = ChunkSequenceOps::delayed::reduce(zipper, add_points);

  //get mean of the points
  double xa = xsum/n;
  double ya = ysum/n;

  //map points onto line by finding the y-intercept and then 
  //also don't instantiate the mapping, just compute them one-by-one so we don't have to make a new sequence on disk
  //so we're iterating through the zipped sequence and finding
  //v = x - xa, the difference of the x coordinate of the point from its mean, this is squared for least-squares distance
  //this is also multiplied by y because it's equivalent to finding the y difference
  auto tmp = ChunkSequenceOps::delayed::map(zipper,[=] (point p) {
    auto [x, y] = p;
    double v = x - xa;
    return point(v * v, v * y);});

  //so now we have a series of points that represent the squared x and y distances to the mean line 
  //so now we're going to add all of these up without instantiating
  auto [Stt, bb] = ChunkSequenceOps::delayed::reduce(tmp, add_points);

  //b = ysum/xsum = slope
  double b = bb / Stt;
  //intercept = mean of y - mean of x * slope
  //this is true because the line must pass through the centroid
  double a = ya - xa * b;
  return point(a, b);
}

}  // namespace ChunkSequenceOps
#endif
