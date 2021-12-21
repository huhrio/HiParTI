#include <ParTI.h>
#include <stdlib.h>
#include "sptensor.h"
#include <string.h>
#include <limits.h>
#include <numa.h>

int sptSparseTensorMulTensor(sptSparseTensor *Z, sptSparseTensor * const X, sptSparseTensor *const Y, sptIndex num_cmodes, sptIndex * cmodes_X, sptIndex * cmodes_Y, int tk, int output_sorting, int placement)
{
	if(experiment_modes == 5){
	    int result;
	    int dram_node;
	    int optane_node;
	    sscanf(getenv("DRAM_NODE"), "%d", &dram_node);
	    sscanf(getenv("OPTANE_NODE"), "%d", &optane_node);
	    int numa_node = dram_node;

	    sptIndex nmodes_X = X->nmodes;
	    sptIndex nmodes_Y = Y->nmodes;
	    sptTimer timer;
	    double total_time = 0;
	    sptNewTimer(&timer, 0);

	    if(num_cmodes >= X->nmodes) {
	        spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
	    }
	    for(sptIndex m = 0; m < num_cmodes; ++m) {
	        if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
	            spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
	        }
	    }

	    sptStartTimer(timer);
	    sptIndex * mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
	    sptIndex ci = nmodes_X - num_cmodes, fi = 0;
	    for(sptIndex m = 0; m < nmodes_X; ++m) {
	        if(sptInArray(cmodes_X, num_cmodes, m) == -1) {
	            mode_order_X[fi] = m;
	            ++ fi;
	        }
	    }
	    sptAssert(fi == nmodes_X - num_cmodes);
	    /// Copy the contract modes while keeping the contraction mode order
	    for(sptIndex m = 0; m < num_cmodes; ++m) {
	        mode_order_X[ci] = cmodes_X[m];
	        ++ ci;
	    }
	    sptAssert(ci == nmodes_X);
	    /// Shuffle tensor indices according to mode_order_X
	    sptSparseTensorShuffleModes(X, mode_order_X);

	    for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
	    sptSparseTensorSortIndex(X, 1, tk);

	    sptStopTimer(timer);
	    double X_time = sptElapsedTime(timer);
	    total_time += X_time;
	    sptStartTimer(timer);

	    unsigned long long tmp_dram_size = 0;
	    FILE *fp;
	    char *s;
	    char path[1035];
	    unsigned long long  i1, i2, i3, i4, i5, i6, i7, i8;
	    fp = popen("numactl -H", "r");
	    while (fgets(path, sizeof(path), fp) != NULL) {
	        s = strstr(path, "node 0 free:");
	        if (s != NULL)
	            if (2 == sscanf(s, "%*[^0123456789]%llu%*[^0123456789]%llu", &i1, &i2)){
	                tmp_dram_size = i2 * 1024 * 1024;
	                //printf("test: %llu B\n", dram_cap);
	                break;
	            }
	    }
	    pclose(fp);

	    unsigned int node_size = sizeof(unsigned long long) + sizeof(unsigned int) + sizeof(unsigned int) + sizeof(unsigned long long*) + sizeof(sptValue*) + sizeof(tensor_node_t*);
	    unsigned long long Y_upper_size = node_size * (Y->nnz + Y->nnz);
	    //printf("%lu\n", Y_upper_size);
	    if (Y_upper_size < tmp_dram_size) numa_set_preferred(dram_node);
	    else numa_set_preferred(numa_node);

	    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
	    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
	    ci = 0;
	    fi = num_cmodes;
	    for(sptIndex m = 0; m < nmodes_Y; ++m) {
	        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) {
	            mode_order_Y[fi] = m;
	            ++ fi;
	        }
	    }
	    sptAssert(fi == nmodes_Y);

	    for(sptIndex m = 0; m < num_cmodes; ++m) {
	        mode_order_Y[ci] = cmodes_Y[m];
	        ++ ci;
	    }
	    sptAssert(ci == num_cmodes);
	    //for(sptIndex m = 0; m < nmodes_Y; ++m)
	    //    printf ("mode_order_Y[m]: %d\n", mode_order_Y[m]);

	    table_t *Y_ht;
	    unsigned int Y_ht_size = Y->nnz;
	    Y_ht = tensor_htCreate(Y_ht_size);

	    omp_lock_t *locks = (omp_lock_t *)malloc(Y_ht_size*sizeof(omp_lock_t));
	    for(size_t i = 0; i < Y_ht_size; i++) {
	        omp_init_lock(&locks[i]);
	    }

	    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
	    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
	    for(sptIndex i = 0; i < num_cmodes;i++){
	        for(sptIndex j = i; j < num_cmodes;j++)
	            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];
	    }
	    //for(sptIndex i = 0; i <= num_cmodes;i++)
	    //    printf("%d ", Y_cmode_inds[i]);
	    //printf("\n");

	    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
	    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
	    //sptIndex* Y_fmode_inds = (sptIndex*) numa_alloc_onnode((Y_num_fmodes + 1) * sizeof(sptIndex), numa_node);
	    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
	    for(sptIndex i = 0; i < Y_num_fmodes;i++){
	        for(sptIndex j = i; j < Y_num_fmodes;j++)
	            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]];
	    }
	    //for(sptIndex i = 0; i <= Y_num_fmodes;i++)
	    //    printf("%d ", Y_fmode_inds[i]);
	    //printf("\n");

	    sptNnzIndex Y_nnz = Y->nnz;
	    unsigned int Y_free_upper = 0;

	#pragma omp parallel for schedule(static) num_threads(tk) shared(Y_ht, Y_num_fmodes, mode_order_Y, num_cmodes, Y_cmode_inds, Y_fmode_inds)
	    for(sptNnzIndex i = 0; i < Y_nnz; i++){
	        if(placement == 3) numa_set_preferred(optane_node);
	        unsigned long long key_cmodes = 0;
	        for(sptIndex m = 0; m < num_cmodes; ++m)
	            key_cmodes += Y->inds[mode_order_Y[m]].data[i] * Y_cmode_inds[m + 1];

	        unsigned long long key_fmodes = 0;
	        for(sptIndex m = 0; m < Y_num_fmodes; ++m)
	            key_fmodes += Y->inds[mode_order_Y[m+num_cmodes]].data[i] * Y_fmode_inds[m + 1];
	        unsigned pos = tensor_htHashCode(key_cmodes);
	        omp_set_lock(&locks[pos]);
	        tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
	        //printf("Y_val.len: %d\n", Y_val.len);
	        unsigned int Y_len = Y_val.len;
	        if(Y_len == 0) {
	            tensor_htInsert(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
	        }
	        else  {
	            tensor_htUpdate(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
	            if (Y_len >= Y_free_upper) Y_free_upper = Y_len + 1;
	            //for(int i = 0; i < Y_val.len; i++)
	            //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]);
	        }
	        omp_unset_lock(&locks[pos]);
	        //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
	    }

	    for(size_t i = 0; i < Y_ht_size; i++) {
	        omp_destroy_lock(&locks[i]);
	    }

	    sptStopTimer(timer);
	    total_time += sptElapsedTime(timer);
	    printf("[Input Processing]: %.2f s\n", sptElapsedTime(timer) + X_time );


	    sptStartTimer(timer);

	    //printf("Sorted X:\n");
	    //sptSparseTensorStatus(X, stdout);
	    //sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
	    //printf("Sorted Y:\n");
	    //sptSparseTensorStatus(Y, stdout);
	    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);

	    /// Set fidx_X: indexing the combined free indices;
	    sptNnzIndexVector fidx_X;
	    //sptStartTimer(timer);
	    /// Set indices for free modes, use X
	    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);

	    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
	    sptIndex *ndims_buf = malloc(nmodes_Z * sizeof *ndims_buf);
	    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
	    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
	        ndims_buf[m] = X->ndims[m];
	    }

	    /// For sorted Y
	    //for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
	    //    ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[m];
	    //}
	    /// For non-sorted Y
	    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
	        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
	    }
	    free(mode_order_X);
	    free(mode_order_Y);

	    // sptSparseTensor *Z_tmp = malloc(tk * sizeof (sptSparseTensor));
	    sptSparseTensor *Z_tmp_dram, *Z_tmp_optane;
	    if(placement == 5) {
	        Z_tmp_dram = numa_alloc_onnode(tk * sizeof (sptSparseTensor), optane_node);
	        Z_tmp_optane = numa_alloc_onnode(tk * sizeof (sptSparseTensor), optane_node);
	    }
	    else{
	        Z_tmp_dram = numa_alloc_onnode(tk * sizeof (sptSparseTensor), dram_node);
	        Z_tmp_optane = numa_alloc_onnode(tk * sizeof (sptSparseTensor), optane_node);
	    }

	    for (int i = 0; i < tk; i++){
	        //result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
	        result = sptNewSparseTensorNuma(&(Z_tmp_dram[i]), nmodes_Z, ndims_buf, dram_node);
	        result = sptNewSparseTensorNuma(&(Z_tmp_optane[i]), nmodes_Z, ndims_buf, optane_node);
	    }

	    //free(ndims_buf);
	    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);

	    unsigned long long dram_cur = 0;
	    unsigned long long dram_cap = 0;
	    unsigned long long Z_mem = 0;

	    fp = popen("numactl -H", "r");  // Open the command for reading
	    while (fgets(path, sizeof(path), fp) != NULL) {  // Read the output a line at a time - output it.
	        s = strstr(path, "node 0 free:");      // Search for string "hassasin" in buff
	        if (s != NULL)                     // If successful then s now points at "hassasin"
	            if (2 == sscanf(s, "%*[^0123456789]%llu%*[^0123456789]%llu", &i1, &i2)){
	                //printf("System DRAM memory: %lu MB\n", i2);
	                dram_cap = i2 * 1024 * 1024 / 1.1; // Should be changed into: memory of the current system - X - Y_ht
	                //printf("test: %llu B\n", dram_cap);
	                break;
	            }
	    }
	    pclose(fp);

	    sptTimer timer_SPA;
	    double time_prep = 0;
	    double time_free_mode = 0;
	    double time_spa = 0;
	    double time_accumulate_z = 0;
	    sptNewTimer(&timer_SPA, 0);

	    // For the progress
	    int fx_counter = fidx_X.len;

	#pragma omp parallel for schedule(static) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Z_tmp_dram, Z_tmp_optane, Y_fmode_inds, Y_ht, Y_cmode_inds, dram_cap, dram_cur, Z_mem, fx_counter)
	    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len - 1; ++fx_ptr) {    // Loop fiber pointers of X
	        int tid = omp_get_thread_num();
	        if(placement == 4) numa_set_preferred(optane_node);
	        fx_counter--;
	        //if (fx_counter % 1000 == 0) printf("Progress: %d\/%d\n", fx_counter, fidx_X.len);
	        if (tid == 0){
	            sptStartTimer(timer_SPA);
	        }
	        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
	        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];

	        /// The total number and memory of SPA for one x fiber.
	        unsigned long long num_SPA_upper = 0;
	        unsigned long long mem_SPA_upper = 0;
	        unsigned long long mem_SPA_cur = 0;
	        bool SPA_in_dram = false;
	        /// The total memory of Z_tmp
	        unsigned long long Z_tmp_mem = 0;
	        /// hashtable size
	        const unsigned int ht_size = 10000;
	        sptIndex nmodes_spa = nmodes_Y - num_cmodes;
	        long int nnz_counter = 0;
	        sptIndex current_idx = 0;

	        /*for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {    // Loop nnzs inside a X fiber
	            sptValue valX = X->values.data[zX];
	            //printf("valX: %f\n", valX);
	            sptIndexVector cmode_index_X;
	            sptNewIndexVector(&cmode_index_X, num_cmodes, num_cmodes);
	            for(sptIndex i = 0; i < num_cmodes; ++i){
	                cmode_index_X.data[i] = X->inds[nmodes_X - num_cmodes + i].data[zX];
	                //printf("\ncmode_index_X[%lu]: %lu\n", i, cmode_index_X.data[i]);
	            }

	            unsigned long long key_cmodes = 0;
	            for(sptIndex m = 0; m < num_cmodes; ++m)
	                key_cmodes += cmode_index_X.data[m] * Y_cmode_inds[m + 1];
	            //printf("key_cmodes: %d\n", key_cmodes);

	            tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
	            //printf("Y_val.len: %d\n", Y_val.len);
	            unsigned int my_len = Y_val.len;
	            if(my_len == 0) continue;
	            num_SPA_upper += my_len;
	        }*/

	        mem_SPA_upper = (Y_free_upper + fx_end - fx_begin) * sizeof(node_t) + sizeof(node_t*) * ht_size + sizeof(table_t);
	        if(mem_SPA_upper + dram_cur <= dram_cap) { // spa in dram
	            dram_cur += mem_SPA_upper;
	            SPA_in_dram = true;
	        }

	        table_t *ht;
	        ht = htCreate(ht_size);
	        mem_SPA_cur = sizeof( node_t*)*ht_size + sizeof( table_t);

	        if (tid == 0){
	            sptStopTimer(timer_SPA);
	            time_prep += sptElapsedTime(timer_SPA);
	        }

	        for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {    // Loop nnzs inside a X fiber
	            if (tid == 0){
	                sptStartTimer(timer_SPA);
	            }
	            sptValue valX = X->values.data[zX];
	            //printf("valX: %f\n", valX);
	            sptIndexVector cmode_index_X;
	            sptNewIndexVector(&cmode_index_X, num_cmodes, num_cmodes);
	            for(sptIndex i = 0; i < num_cmodes; ++i){
	                cmode_index_X.data[i] = X->inds[nmodes_X - num_cmodes + i].data[zX];
	                //printf("\ncmode_index_X[%lu]: %lu\n", i, cmode_index_X.data[i]);
	            }

	            unsigned long long key_cmodes = 0;
	            for(sptIndex m = 0; m < num_cmodes; ++m)
	                key_cmodes += cmode_index_X.data[m] * Y_cmode_inds[m + 1];
	            //printf("key_cmodes: %d\n", key_cmodes);

	            tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
	            //printf("Y_val.len: %d\n", Y_val.len);
	            unsigned int my_len = Y_val.len;
	            if (tid == 0){
	                sptStopTimer(timer_SPA);
	                time_free_mode += sptElapsedTime(timer_SPA);
	            }
	            if(my_len == 0) continue;

	            if (tid == 0){
	                sptStartTimer(timer_SPA);
	            }
	            if(placement == 4) numa_set_preferred(optane_node);
	            for(int i = 0; i < my_len; i++){
	                unsigned long long fmode =  Y_val.key_FM[i];
	                //printf("i: %d, Y_val.key_FM[i]: %lu, Y_val.val[i]: %f\n", i, Y_val.key_FM[i], Y_val.val[i]);
	                sptValue spa_val = htGet(ht, fmode);
	                float result = Y_val.val[i] * valX;
	                if(spa_val == LONG_MIN) {
	                    htInsert(ht, fmode, result);
	                    mem_SPA_cur += sizeof(node_t);
	                    nnz_counter++;
	                }
	                else
	                    htUpdate(ht, fmode, spa_val + result);
	            }

	            if (tid == 0){
	                sptStopTimer(timer_SPA);
	                time_spa += sptElapsedTime(timer_SPA);
	            }

	        }

	        if (tid == 0){
	            sptStartTimer(timer_SPA);
	        }

	        if(SPA_in_dram) dram_cur = dram_cur - mem_SPA_upper + mem_SPA_cur;
	        Z_tmp_mem = nnz_counter * (nmodes_Z * sizeof(sptIndex) + sizeof(sptValue));
	        Z_mem += Z_tmp_mem;


	        if(Z_tmp_mem + dram_cur <= dram_cap && (tid % 7 != 0)){
	            dram_cur += Z_tmp_mem;
	            for(int i = 0; i < ht->size; i++){
	                if (placement == 5 && fx_ptr%(ht_size/10) == 0) numa_set_preferred(optane_node);
	                node_t *temp = ht->list[i];
	                while(temp){
	                    unsigned long long idx_tmp = temp->key;
	                    //nnz_counter++;
	                    for(sptIndex m = 0; m < nmodes_spa; ++m) {
	                        //sptAppendIndexVector(&Z_tmp_dram[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
	                        sptAppendIndexVectorNuma(&Z_tmp_dram[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
	                    }
	                    //printf("val: %f\n", temp->val);
	                    //sptAppendValueVector(&Z_tmp_dram[tid].values, temp->val);
	                    sptAppendValueVectorNuma(&Z_tmp_dram[tid].values, temp->val);
	                    node_t* pre = temp;
	                    temp = temp->next;
	                    free(pre);
	                    //numa_free(pre, sizeof(node_t));
	                }
	            }
	            Z_tmp_dram[tid].nnz += nnz_counter;
	            for(sptIndex i = 0; i < nnz_counter; ++i) {
	                for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
	                    //sptAppendIndexVector(&Z_tmp_dram[tid].inds[m], X->inds[m].data[fx_begin]);
	                    sptAppendIndexVectorNuma(&Z_tmp_dram[tid].inds[m], X->inds[m].data[fx_begin]);
	                }
	            }
	        }
	        else{
	            for(int i = 0; i < ht->size; i++){
	                if (placement == 5 && fx_ptr%(ht_size/10) == 0) numa_set_preferred(optane_node);
	                node_t *temp = ht->list[i];
	                while(temp){
	                    unsigned long long idx_tmp = temp->key;
	                    //nnz_counter++;
	                    for(sptIndex m = 0; m < nmodes_spa; ++m) {
	                        //sptAppendIndexVector(&Z_tmp_optane[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
	                        sptAppendIndexVectorNuma(&Z_tmp_optane[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
	                    }
	                    //printf("val: %f\n", temp->val);
	                    //sptAppendValueVector(&Z_tmp_optane[tid].values, temp->val);
	                    sptAppendValueVectorNuma(&Z_tmp_optane[tid].values, temp->val);
	                    node_t* pre = temp;
	                    temp = temp->next;
	                    free(pre);
	                    //numa_free(pre, sizeof(node_t));
	                }
	            }
	            Z_tmp_optane[tid].nnz += nnz_counter;
	            for(sptIndex i = 0; i < nnz_counter; ++i) {
	                for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
	                    //sptAppendIndexVector(&Z_tmp_optane[tid].inds[m], X->inds[m].data[fx_begin]);
	                    sptAppendIndexVectorNuma(&Z_tmp_optane[tid].inds[m], X->inds[m].data[fx_begin]);
	                }
	            }
	        }
	        htFree(ht);
	        if(SPA_in_dram) dram_cur -= mem_SPA_cur;

	        if (tid == 0){
	            sptStopTimer(timer_SPA);
	            time_accumulate_z += sptElapsedTime(timer_SPA);
	        }
	        //printf("Z:\n");
	        //sptDumpSparseTensor(Z, 0, stdout);
	    }   // End Loop fiber pointers of X

	    //sptAssert(sptDumpSparseTensor(Z, 0, stdout) == 0);

	    sptStopTimer(timer);
	    double main_computation = sptElapsedTime(timer);
	    total_time += main_computation;
	    double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
	    printf("[Index Search]: %.2f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
	    printf("[Accumulation]: %.2f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);

	    sptStartTimer(timer);
	    if(Z_mem + dram_cur < dram_cap) numa_node = dram_node;

	    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
	    unsigned long long Z_total_size = 0;

	    Z_tmp_start[0] = 0;
	    for(int i = 0; i < tk; i++){
	        Z_tmp_start[i + 1] = Z_tmp_dram[i].nnz + Z_tmp_optane[i].nnz +  Z_tmp_start[i];
	        Z_total_size +=  Z_tmp_dram[i].nnz + Z_tmp_optane[i].nnz;
	        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
	    }

	    if(placement == 6) {
	        result = sptNewSparseTensorWithSizeNuma(Z, nmodes_Z, ndims_buf, optane_node, Z_total_size);
	    }
	    else{
	        result = sptNewSparseTensorWithSizeNuma(Z, nmodes_Z, ndims_buf, numa_node, Z_total_size);
	    }
	    //result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size);

	#pragma omp parallel for schedule(static) num_threads(tk) shared(Z_tmp_dram, Z_tmp_optane, Z, nmodes_Z, Z_tmp_start)
	    for(int i = 0; i < tk; i++){
	        int tid = omp_get_thread_num();
	        if(Z_tmp_dram[tid].nnz > 0){
	            for(sptIndex m = 0; m < nmodes_Z; ++m)
	                sptAppendIndexVectorWithVectorStartFromNuma(&Z->inds[m], &Z_tmp_dram[tid].inds[m], Z_tmp_start[tid]);
	            sptAppendValueVectorWithVectorStartFromNuma(&Z->values, &Z_tmp_dram[tid].values, Z_tmp_start[tid]);
	        }
	        if(Z_tmp_optane[tid].nnz > 0){
	            for(sptIndex m = 0; m < nmodes_Z; ++m)
	                sptAppendIndexVectorWithVectorStartFromNuma(&Z->inds[m], &Z_tmp_optane[tid].inds[m], Z_tmp_start[tid] + Z_tmp_dram[tid].nnz);
	            sptAppendValueVectorWithVectorStartFromNuma(&Z->values, &Z_tmp_optane[tid].values, Z_tmp_start[tid] + Z_tmp_dram[tid].nnz);
	        }
	    }

	    sptStopTimer(timer);

	    total_time += sptPrintElapsedTime(timer, "Writeback");
	    sptStartTimer(timer);

	    sptSparseTensorSortIndex(Z, 1, tk);

	    sptStopTimer(timer);
	    total_time += sptPrintElapsedTime(timer, "Output Sorting");
	    printf("[Total time]: %.2f s\n", total_time);
	    //system("numactl -H");
	    printf("\n");
	}

	return 0;
}
