//Let's try writing our own group_by method to make sure I actually understand what I'm doing here.

//basic logic:
//read all the stuff in, we assign into a series of buckets in a vector based on the passed ID since we'll be reading this concurrently with the values

//seems simple enough, but it's a bit tricky to implement. kind of a nice example for the library, though.


chunk_seq& group_by_index(const chunk_seq& seq, const chunk_seq& ids, std::vector<chunk_seq>& externalSequenceVector, const std::string& result_prefix = "bucket"){

  const size_t num_buckets = externalSequenceVector.size();
  const size_t ept = CHUNK_SIZE / sizeof(T);
  const size_t num_drives = GetSSDList().size();

    //there exists one file per drive ideally so we don't have too many fds open
    //the first thing we need to do is actually clear the drives of any stale data, so we'll open them with O_TRUNC

std::vector<std::string> filenames(num_drives);
for(int i = 0; i < num_drives; i++){
filesnames[i] = GetFileName(result_prefix, i);
int fd = open(filenames[i], O_WRONLY | O_CREAT | O_TRUNC, 0644);
SYSCALL(fd); //syscall just checks whether the open worked correctly 
close(fd);
SYSCALL(fd);
}
//next we open our writer
  UnorderedWriterConfig wcfg;
  wcfg.num_threads = num_drives;
  wcfg.io_uring_size = 32;
  wcfg.queue_size = 64;
  wcfg.num_files = num_drives;
  UnorderedFileWriter<T> writer;
  writer.Start(filenames, wcfg);


  //now we need to allocate the buffers -- this should be able to be done in parallel

std::vector<T*> buffers(num_drives);
std::vector<size_t> buffer_counters(num_buckets, 0);
parlay::parallel_for(0, num_drives, [&](long i){

    buffers[i] = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);


});

//yeah I'm not writing the coalesce and flush logic so this is copied over. I will comment it though
  auto flush = [&](size_t b) { //
    const size_t d = slot++ % num_drives;
    const size_t base = drive_off[d];
    drive_off[d] += CHUNK_SIZE;
    const size_t used = buffer_counters[b];

    if (used < ept) memset(buffers[b] + used, 0, (ept - used) * sizeof(T));
    externalSequenceVector[b].chunks.push_back(
        chunk{filenames[d], base, used * sizeof(T),
              externalSequenceVector[b].chunks.size()});

    writer.Push(std::shared_ptr<T>(buffers[b], free), ept, d, base);
    buffers[b] = (T*)aligned_alloc(O_DIRECT_MEMORY_ALIGNMENT, CHUNK_SIZE);
    CHECK(buffers[b] != nullptr) << "count_sort: buffer alloc failed";
    buffer_counters[b] = 0;
  };


  Nreader reader;
  reader.prep({ids&, seq&});
  reader.start();



  while(true){

    auto i = reader.poll(); //important note: i.ptrs[0] is now the ids seq pointer, i.ptrs[1] is now the seq seq pointer
    if(!.valid()){
        break;
    }
    T* id_pointer = i.ptrs[0];
    T* seq_pointer = i.ptrs[1];
    size_t num_elements = i.sizes[0]; //this is the (matching) number of actual T elements

    for(int k = 0; k < num_elements; k++){//this is seqeuntial to avoid the buffer write conflicts
        int bucket_id = id_pointer[k];
        buffers[bucket_id][bucket_counters++] = seq_pointer[k];
        if(bucket_counters[bucket_id] == CHUNK_SIZE/sizeof(T)) //if bucket is full
        {
            flush(bucket_id);
        }



    }
  //otherwise there's stuff to read in the respective pair block i
    reader.free(i);

    //this is all well and good if the buckets are all flushed at the end

    
    

  }
  for(int i = 0; i < num_drives; i++){
    if(bucket_counters[i] > 0){
    flush(i);
    }
    free(buffers[i]);
  }
writer.Wait();
}