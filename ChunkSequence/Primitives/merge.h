
//this file is not yet of any practical use, but maybe we want to implement a proper merge operation later on
#ifndef EXTERNAL_MERGE_H
#define EXTERNAL_MERGE_H




namespace plaid {


//NOT currently functional
//returns a new, merged list of chunks from two external sequences
//the goal here is actually to partition multiple times about the median chunks
chunk_seq merge(chunk_seq& seq1, chunk_seq& seq2){


if()

auto median_chunk_seq1 = seq1.chunks[std::floor(seq1.chunks.size()/2)];

auto median_chunk_seq2 = seq2.chunks[std::floor(seq2.chunks.size()/2)];

//we're just going to use a linear search approach for right now, can test it against scan_find later on

//now find the median elements

auto med1 = plaid::single_chunk_ops::find(median_chunk_seq1, median_chunk_seq1.used/sizeof(T));
auto index_of_med1 = median_chunk_seq1.index * CHUNKSIZE + median_chunk_seq1.used();

auto med2= plaid::single_chunk_ops::find(median_chunk_seq1, median_chunk_seq2.used/sizeof(T));
auto index_of_med2 = median_chunk_seq2.index * CHUNKSIZE + median_chunk_seq2.used();

if(med1 < med2){
    auto med3 = med2;
    med2= med1;
    med1 = med3;
    index_of_med1 = index_of_med2;
}

//now we know that med1 is the larger of the two medians
//binary searching across chunks is quite costly, though

size_t index_max_containing = chunk_binary_search(med1)

parlay::par_do{
    //maybe make these cuts delayed at some point
plaid::seq_merge(seq1.cut(0, index_of_med1), seq2.cut(0, index_max_containing));
plaid::seq_merge(seq1.cut(index_of_med1, size1.size()), seq2.cut(index_max_containing, seq2.size()));

}

}



std::pair<std::vector<chunk_seq>, std::vector<chunk_seq>> cut_by_chunks(chunk_seq seq, size_t index){

auto cut1(chunk_seq.begin(), chunk_seq.begin()+index);
auto cut2(chunk_seq.begin()+index, chunk_seq.end());

return std::pair<cut1, cut2>;

}




}  // namespace plaid



namespace single_chunk_ops{

template<typename T>
T find(chunk p, size_t index){ //the  SK hynix Platinum P41 NVMe SSD and most commercial SSDs can't actually read any smaller than a 512 kb block size

int fd = open(p.filename, O_DIRECT | O_RDONLY, 0644);

T buffer[CHUNK_SIZE/sizeof(T)];

read(fd, buffer, CHUNK_SIZE); //this is going to be rounded up anyway so we might as well read the whole chunk


return buffer[index];


close(fd);

}

}

#endif