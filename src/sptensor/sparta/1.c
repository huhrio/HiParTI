#include <ParTI.h>
#include <stdlib.h>
#include "sptensor.h"
#include <string.h>
#include <limits.h>
#include <numa.h>

int sptSparseTensorMulTensor(sptSparseTensor *Z, sptSparseTensor * const X, sptSparseTensor *const Y, sptIndex num_cmodes, sptIndex * cmodes_X, sptIndex * cmodes_Y, int tk, int output_sorting, int placement)
{
	//1: COOY + HTA
	if(experiment_modes == 1){
	    int result;
	    /// The number of threads
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
	    /// Shuffle X indices and sort X as the order of free modes -> contract modes; mode_order also separate all the modes to free and contract modes separately.
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
	    // printf("Permuted X:\n");
	    // sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
	    for(sptIndex m = 0; m < nmodes_X; ++m) mode_order_X[m] = m; // reset mode_order
	    sptSparseTensorSortIndex(X, 1, tk);

	    sptStopTimer(timer);
	    double X_time = sptElapsedTime(timer);
	    total_time += X_time;
	    sptStartTimer(timer);

	    /// Shuffle Y indices and sort Y as the order of free modes -> contract modes
	    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);
	    sptIndex * mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
	    ci = 0;
	    fi = num_cmodes;
	    for(sptIndex m = 0; m < nmodes_Y; ++m) {
	        if(sptInArray(cmodes_Y, num_cmodes, m) == -1) { // m is not a contraction mode
	            mode_order_Y[fi] = m;
	            ++ fi;
	        }
	    }
	    sptAssert(fi == nmodes_Y);
	    /// Copy the contract modes while keeping the contraction mode order
	    for(sptIndex m = 0; m < num_cmodes; ++m) {
	        mode_order_Y[ci] = cmodes_Y[m];
	        ++ ci;
	    }
	    sptAssert(ci == num_cmodes);
	    /// Shuffle tensor indices according to mode_order_Y
	    sptSparseTensorShuffleModes(Y, mode_order_Y);
	    // printf("Permuted Y:\n");
	    for(sptIndex m = 0; m < nmodes_Y; ++m) mode_order_Y[m] = m; // reset mode_order
	    sptSparseTensorSortIndex(Y, 1, tk);
	    sptStopTimer(timer);
	    total_time += sptElapsedTime(timer);
	    printf("[Input Processing]: %.6f s\n", X_time + sptElapsedTime(timer));

	    //printf("Sorted X:\n");
	    //sptAssert(sptDumpSparseTensor(X, 0, stdout) == 0);
	    //printf("Sorted Y:\n");
	    //sptAssert(sptDumpSparseTensor(Y, 0, stdout) == 0);

	    /// Set fidx_X: indexing the combined free indices and fidx_Y: indexing the combined contract indices
	    sptNnzIndexVector fidx_X, fidx_Y;
	    //sptStartTimer(timer);
	    /// Set indices for free modes, use X
	    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);
	    /// Set indices for contract modes, use Y
	    sptSparseTensorSetIndices(Y, mode_order_Y, num_cmodes, &fidx_Y);
	    //sptStopTimer(timer);
	    //sptPrintElapsedTime(timer, "Set fidx X,Y");
	    //sptPrintElapsedTime(timer, "Set fidx X");
	    //printf("fidx_X: \n");
	    //sptDumpNnzIndexVector(&fidx_X, stdout);
	    //printf("fidx_Y: \n");
	    //sptDumpNnzIndexVector(&fidx_Y, stdout);
	    free(mode_order_X);
	    free(mode_order_Y);

	    /// Allocate the output tensor
	    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
	    sptIndex *ndims_buf = malloc(nmodes_Z * sizeof *ndims_buf);
	    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
	    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
	        ndims_buf[m] = X->ndims[m];
	    }
	    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
	        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[m];
	    }
	    /// Each thread with a local Z_tmp
	    sptSparseTensor *Z_tmp = malloc(tk * sizeof (sptSparseTensor));

	    for (int i = 0; i < tk; i++){
	        result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
	    }

	    //free(ndims_buf);
	    spt_CheckError(result, "CPU  SpTns * SpTns", NULL);

	    sptTimer timer_SPA;
	    double time_prep = 0;
	    double time_free_mode = 0;
	    double time_spa = 0;
	    double time_accumulate_z = 0;

	    sptNewTimer(&timer_SPA, 0);

	    sptStartTimer(timer);

	    // For the progress
	    int fx_counter = fidx_X.len;

	#pragma omp parallel for schedule(static) num_threads(tk) shared(fidx_X, fidx_Y, nmodes_X, nmodes_Y, num_cmodes, Z_tmp, fx_counter)
	    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len - 1; ++fx_ptr) {    // Loop fiber pointers of X
	        int tid = omp_get_thread_num();
	        //Print the progress
	        fx_counter--;
	        //if (fx_counter % 1 == 0) printf("Progress: %d\/%d\n", fx_counter, fidx_X.len);
	        if (tid == 0){
	            sptStartTimer(timer_SPA);
	        }

	        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
	        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];
	        sptIndex nmodes_spa = nmodes_Y - num_cmodes;
	        long int nnz_counter = 0;

	        /// Calculate key range for hashtable
	        sptIndex* inds_buf = (sptIndex*)malloc((nmodes_spa + 1) * sizeof(sptIndex));
	        sptIndex current_idx = 0;
	        for(sptIndex i = 0; i < nmodes_spa + 1; i++) inds_buf[i] = 1;
	        for(sptIndex i = 0; i < nmodes_spa;i++){
	            for(sptIndex j = i; j < nmodes_spa;j++)
	                inds_buf[i] = inds_buf[i] * Y->ndims[j + num_cmodes];
	        }

	        /// Create a hashtable for SPAs
	        table_t *ht;
	        const unsigned int ht_size = 10000;
	        ht = htCreate(ht_size);

	        if (tid == 0){
	            sptStopTimer(timer_SPA);
	            time_prep += sptElapsedTime(timer_SPA);
	        }

	        /// zX has common free indices
	        for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {    // Loop nnzs inside a X fiber
	            if (tid == 0){
	                sptStartTimer(timer_SPA);
	            }

	            sptValue valX = X->values.data[zX];
	            sptIndexVector cmode_index_X;
	            sptNewIndexVector(&cmode_index_X, num_cmodes, num_cmodes);
	            for(sptIndex i = 0; i < num_cmodes; ++i){
	                 cmode_index_X.data[i] = X->inds[nmodes_X - num_cmodes + i].data[zX];
	                 //printf("\ncmode_index_X[%lu]: %lu", i, cmode_index_X[i]);
	             }

	            sptNnzIndex fy_begin = -1;
	            sptNnzIndex fy_end = -1;

	            for(sptIndex j = 0; j < fidx_Y.len; j++){
	                for(sptIndex i = 0; i< num_cmodes; i++){
	                    if(cmode_index_X.data[i] != Y->inds[i].data[fidx_Y.data[j]]) break;
	                    if(i == (num_cmodes - 1)){
	                        fy_begin = fidx_Y.data[j];
	                        fy_end = fidx_Y.data[j+1];
	                        break;
	                    }
	                    //printf("\ni: %lu, current_idx: %lu, Y->inds[i].data[fidx_Y.data[current_idx]]: %lu\n", i, current_idx, Y->inds[i].data[fidx_Y.data[current_idx]]);
	                }
	                if (fy_begin != -1 || fy_end != -1) break;
	            }

	            if (tid == 0){
	                sptStopTimer(timer_SPA);
	                time_free_mode += sptElapsedTime(timer_SPA);
	            }

	            if (fy_begin == -1 || fy_end == -1) continue;
	            //printf("zX: %lu, valX: %.2f, cmode_index_X[0]: %u, zY: [%lu, %lu]\n", zX, valX, cmode_index_X.data[0], fy_begin, fy_end);

	            if (tid == 0) sptStartTimer(timer_SPA);

	            /// zY has common contraction indices
	            for(sptNnzIndex zY = fy_begin; zY < fy_end; ++ zY) {    // Loop nnzs inside a Y fiber
	                long int tmp_key = 0;
	                for(sptIndex m = 0; m < nmodes_spa; ++m)
	                    tmp_key += Y->inds[m + num_cmodes].data[zY] * inds_buf[m + 1];
	                sptValue val = htGet(ht, tmp_key);
	                if(val == LONG_MIN)
	                    htInsert(ht, tmp_key, Y->values.data[zY] * valX);
	                else
	                    htUpdate(ht, tmp_key, val + (Y->values.data[zY] * valX));
	                //printf("val: %f\n", val);
	            }
	            if (tid == 0){
	                sptStopTimer(timer_SPA);
	                time_spa += sptElapsedTime(timer_SPA);
	            }

	        }   // End Loop nnzs inside a X fiber

	        if (tid == 0){
	            sptStartTimer(timer_SPA);
	        }

	        /// Write back to Z
	        for(int i = 0; i < ht->size; i++){
	            node_t *temp = ht->list[i];
	            while(temp){
	                long int idx_tmp = temp->key;
	                nnz_counter++;
	                for(sptIndex m = 0; m < nmodes_spa; ++m) {
	                    //printf("idx_tmp: %lu, m: %d, (idx_tmp inds_buf[m])/inds_buf[m+1]): %d\n", idx_tmp, m, (idx_tmp%inds_buf[m])/inds_buf[m+1]);
	                    sptAppendIndexVector(&Z_tmp[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%inds_buf[m])/inds_buf[m+1]);
	                }
	                //printf("val: %f\n", temp->val);
	                sptAppendValueVector(&Z_tmp[tid].values, temp->val);
	                node_t* pre = temp;
	                temp = temp->next;
	                free(pre);
	            }
	        }

	        Z_tmp[tid].nnz += nnz_counter;
	        for(sptIndex i = 0; i < nnz_counter; ++i) {
	            for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
	                sptAppendIndexVector(&Z_tmp[tid].inds[m], X->inds[m].data[fx_begin]);
	            }
	        }

	        // release spa hashtable
	        htFree(ht);

	        if (tid == 0){
	            sptStopTimer(timer_SPA);
	            time_accumulate_z += sptElapsedTime(timer_SPA);
	        }
	    }   // End Loop fiber pointers of X

	sptStopTimer(timer);
	double main_computation = sptElapsedTime(timer);
	total_time += main_computation;
	double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
	printf("[Index Search]: %.2f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
	printf("[Accumulation]: %.2f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);

	sptStartTimer(timer);

	/// Append Z_tmp to Z
	    //Calculate the indecies of Z
	    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
	    unsigned long long Z_total_size = 0;

	    Z_tmp_start[0] = 0;
	    for(int i = 0; i < tk; i++){
	        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
	        Z_total_size +=  Z_tmp[i].nnz;
	    }
	    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size);

	#pragma omp parallel for schedule(static) num_threads(tk) shared(Z, nmodes_Z, Z_tmp_start)
	    for(int i = 0; i < tk; i++){
	        int tid = omp_get_thread_num();
	        if(Z_tmp[tid].nnz > 0){
	            for(sptIndex m = 0; m < nmodes_Z; ++m)
	                sptAppendIndexVectorWithVectorStartFromNuma(&Z->inds[m], &Z_tmp[tid].inds[m], Z_tmp_start[tid]);
	            sptAppendValueVectorWithVectorStartFromNuma(&Z->values, &Z_tmp[tid].values, Z_tmp_start[tid]);
	        }
	    }

	    sptStopTimer(timer);
	    total_time += sptPrintElapsedTime(timer, "Writeback");
	    sptStartTimer(timer);

	    sptSparseTensorSortIndex(Z, 1, tk);

	    sptStopTimer(timer);
	    total_time += sptPrintElapsedTime(timer, "Output Sorting");
	    printf("[Total time]: %.6f s\n", total_time);
	    printf("\n");
	}
}
