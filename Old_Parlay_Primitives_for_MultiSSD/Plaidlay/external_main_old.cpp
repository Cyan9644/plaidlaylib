#include "externalSeq.h"

auto add = [](size_t a, size_t b) {return a + b;};

void mapThroughput();

int main() {
    // mapThroughput();
    // later on we could use macros to enforce file name matching identifier?
    // externalSeq<size_t> nums = externalSeqOps::randPerm<size_t>("nums", 24);
    // // TODO: for some reason I need to provide that both are long long for the generic here otherwise I get negatives, probably overflow
    // // okay size_t just works for some reason here
    // std::cout << externalSeqOps::reduce<>(nums, add, (size_t)0) << std::endl;
    // externalSeq<size_t> halved = externalSeqOps::map<size_t, size_t>(nums, "halved", [](size_t x) { return x / 2; });
    // std::cout << externalSeqOps::reduce<>(halved, add, (size_t)0) << std::endl;
    // externalSeq<size_t> modTen = externalSeqOps::filter<>(nums, "modTen", [](size_t a) {return a % 10 == 0;});
    // std::cout << externalSeqOps::reduce<>(modTen, add, (size_t)0) << std::endl;

    externalSeq<size_t> nums2 = externalSeqOps::randPerm<size_t>("nums", 24);

    std::cout << externalSeqOps::reduce<>(nums2, add, (size_t)0) << std::endl;
    externalSeq<size_t> halved2 = externalSeqOps::map<size_t, size_t>(nums2, "halved", [](size_t x) { return x / 2; });
    std::cout << externalSeqOps::reduce<>(halved2, add, (size_t)0) << std::endl;
    externalSeq<size_t> scanner = externalSeqOps::naiveScan<>(nums2, (std::string) "scan");
    externalSeqOps::check_in_mem(scanner, (std::string) "scan");
    // std::cout << externalSeqOps::reduce<>(modTen, add, (size_t)0) << std::endl;


    return 0;
}




void mapThroughput() {
    parlay::internal::timer timer("Map");
    timer.next("Start prep");
    externalSeq<size_t> nums = externalSeqOps::randPerm<size_t>("nums", 24);
    timer.next("Start map");
    auto result = externalSeqOps::reduce<>(nums, add, (size_t)0);
    double time = timer.next_time();
    double throughput = GetThroughput(nums.files, time);
    std::cout << "throughput is " << throughput << " GB per sec" <<  std::endl;
}