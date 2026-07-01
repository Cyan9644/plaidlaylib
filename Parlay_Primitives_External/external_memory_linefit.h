#include <utility>                               // Includes the C++ standard utility library, required for using std::pair.

#include <parlay/primitives.h>                   // Includes ParlayLib's core parallel primitives, such as parlay::reduce.
#include <parlay/delayed.h>                      // Includes ParlayLib's delayed sequences for lazy evaluation (parlay::delayed::map).

// ************************************************************** // Original block comment top border.
// Fits a set of points to a line minimizing chi-squared.         // Explains the mathematical purpose: linear regression/best-fit line.
// Returns the y intercept at x=0 and the slope.                  // Describes the expected output (a pair containing intercept and slope).
// Parallel version of the "fit" algorithm from:                  // Notes that this utilizes parallel computation to speed up math.
// "Numerical Recipes: The art of scientific computing"           // Credits the foundational text for the algorithm's math.
// by Press, Teukolsky, Vetterling, and Flannery, section 15.2.   // Credits the authors and precise section for reference.
// ************************************************************** // Original block comment bottom border.

using point = std::pair<double,double>;          // Defines a type alias 'point' as a pair of double-precision floating-point numbers (x, y).

// a binary associative operator that elementwise adds two points // Original comment describing the lambda function below.
auto f = [] (point a, point b) {                 // Declares a lambda function 'f' that takes two 'point' variables ('a' and 'b').
  return point{a.first + b.first, a.second + b.second};}; // Returns a new point containing the sum of their x and y values, respectively.
auto add_points = parlay::binary_op(f, point(0,0)); // Wraps lambda 'f' and an identity element (0,0) into a ParlayLib binary operator for parallel reduction.

// The algorithm                                 // Original comment denoting the start of the core logic.
template <class Seq>                             // Template declaration, allowing the function to accept any ParlayLib-compatible sequence container.
auto linefit(const Seq& points) {                // Defines the function 'linefit', which takes a read-only reference to a sequence of points.
  long n = points.size();                        // Retrieves the total number of points in the dataset.
  auto [xsum, ysum] = reduce(points, add_points); // Runs a parallel reduction tree to sum all x's and all y's, unpacking them into xsum and ysum.
  double xmean = xsum/n;                            // Computes the arithmetic mean (average) of all the x coordinates.
  double ymean = ysum/n;                            // Computes the arithmetic mean (average) of all the y coordinates.
  auto tmp = parlay::delayed::map(points,[=] (point p) {  // Lazily applies a lambda to every point in parallel. '[=]' captures 'xmean' by value.
    auto [x, y] = p;                             // Unpacks the current point 'p' into its constituent 'x' and 'y' coordinates.
    double v = x - xmean;                           // Calculates the deviation of the current x coordinate from the mean of x.
    return point(v * v, v * y);});               // Returns a pair representing (x - mean(x))^2 and (x - mean(x)) * y.
  auto [Stt, bb] = parlay::reduce(tmp, add_points); // Runs another parallel reduction on the lazy map 'tmp' to sum the squared deviations and cross-terms.
  double b = bb / Stt;                           // Calculates the slope 'b' of the best-fit line (sum of cross terms / sum of squares).
  double a = ymean - xmean * b;                        // Calculates the y-intercept 'a' using the point-slope form and the centroid (xmean, ymean).
  return point(a, b);                            // Returns the computed y-intercept and slope as a single 'point' pair.
}                                                // Closes the linefit function block.

//in_memory linefit
//breaks a point into x,y
using point = std::pair<double,double>;

// a binary associative operator that elementwise adds two points
auto f = [] (point a, point b) {
  return point{a.first + b.first, a.second + b.second};};

auto add_points = parlay::binary_op(f, point(0,0));

// The algorithm
template <class Seq>
auto linefit(const Seq& points) {

 using point = std::pair<double,double>;

// a binary associative operator that elementwise adds two points
auto f = [] (point a, point b) {
  return point{a.first + b.first, a.second + b.second};};
auto add_points = parlay::binary_op(f, point(0,0));

// The algorithm
template <class Seq>
auto linefit(const Seq& points) {
  long n = points.size();
  auto [xsum, ysum] = reduce(points, add_points);
  double xa = xsum/n;
  double ya = ysum/n;
  auto tmp = delayed_map(points,[=] (point p) {
    auto [x, y] = p;
    double v = x - xa;
    return point(v * v, v * y);}); //this is an optimization but it is essentially just going to compute the mean dist 

  auto [Stt, bb] = reduce(tmp, add_points);
  double b = bb / Stt;
  double a = ya - xa * b;
  return point(a, b);
}