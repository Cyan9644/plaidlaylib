#include <pthread.h>
#include "plaidlay.h"
#include <cassert>
#include <math.h>
#include <iostream>
#include <fcntl.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <liburing.h>
#include <cstring>
#include <parlay/parallel.h>
#include <parlay/primitives.h>


template <typename T>
class bitvector {

};