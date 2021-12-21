#include <ParTI.h>
#include <stdlib.h>
#include "sptensor.h"
#include <string.h>
#include <limits.h>
#include <numa.h>

int sptSparseTensorMulTensor(sptSparseTensor *Z, sptSparseTensor * const X, sptSparseTensor *const Y, sptIndex num_cmodes, sptIndex * cmodes_X, sptIndex * cmodes_Y, int tk, int output_sorting, int placement)
{
	//3: HTY + HTA
		if(experiment_modes == 3){
		    int result;
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
		    // sptSparseTensorSortIndexCmode(X, 1, 1, 1, 2);
		    sptSparseTensorSortIndex(X, 1, tk);

		    sptStopTimer(timer);
		    //total_time += sptPrintElapsedTime(timer, "Sort X");
		    double X_time = sptElapsedTime(timer);
		    total_time += X_time;
		    sptStartTimer(timer);

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
		    /// Copy the contract modes while keeping the contraction mode order
		    for(sptIndex m = 0; m < num_cmodes; ++m) {
		        mode_order_Y[ci] = cmodes_Y[m];
		        ++ ci;
		    }

		    table_t *Y_ht;
		    unsigned int Y_ht_size = Y->nnz;
		    Y_ht = tensor_htCreate(Y_ht_size);

		    omp_lock_t *locks = (omp_lock_t *)malloc(Y_ht_size*sizeof(omp_lock_t));
		    for(size_t i = 0; i < Y_ht_size; i++) omp_init_lock(&locks[i]);

		    sptIndex* Y_cmode_inds = (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
		    for(sptIndex i = 0; i < num_cmodes + 1; i++) Y_cmode_inds[i] = 1;
		    for(sptIndex i = 0; i < num_cmodes;i++){
		        for(sptIndex j = i; j < num_cmodes;j++)
		            Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];
		    }

		    sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
		    sptIndex* Y_fmode_inds = (sptIndex*)malloc((Y_num_fmodes + 1) * sizeof(sptIndex));
		    for(sptIndex i = 0; i < Y_num_fmodes + 1; i++) Y_fmode_inds[i] = 1;
		    for(sptIndex i = 0; i < Y_num_fmodes;i++){
		        for(sptIndex j = i; j < Y_num_fmodes;j++)
		            Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]];
		    }

		    sptNnzIndex Y_nnz = Y->nnz;
		#pragma omp parallel for schedule(static) num_threads(tk) shared(Y_ht, Y_num_fmodes, mode_order_Y, num_cmodes, Y_cmode_inds, Y_fmode_inds)
		    for(sptNnzIndex i = 0; i < Y_nnz; i++){
		        //if (Y->values.data[i] <0.00000001) continue;
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
		        if(Y_val.len == 0) {
		            tensor_htInsert(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
		        }
		        else  {
		            tensor_htUpdate(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
		            //for(int i = 0; i < Y_val.len; i++)
		            //    printf("key_FM: %lu, Y_val: %f\n", Y_val.key_FM[i], Y_val.val[i]);
		        }
		        omp_unset_lock(&locks[pos]);
		        //sprintf("i: %d, key_cmodes: %lu, key_fmodes: %lu\n", i, key_cmodes, key_fmodes);
		    }

		    for(size_t i = 0; i < Y_ht_size; i++) omp_destroy_lock(&locks[i]);

		    sptStopTimer(timer);
		    total_time += sptElapsedTime(timer);
		    printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer) + X_time);

		    sptNnzIndexVector fidx_X;
		    /// Set indices for free modes, use X
		    sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, &fidx_X);
		    //printf("fidx_X: \n");
		    //sptDumpNnzIndexVector(&fidx_X, stdout);

		    /// Allocate the output tensor
		    sptIndex nmodes_Z = nmodes_X + nmodes_Y - 2 * num_cmodes;
		    sptIndex *ndims_buf = malloc(nmodes_Z * sizeof *ndims_buf);
		    spt_CheckOSError(!ndims_buf, "CPU  SpTns * SpTns");
		    for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
		        ndims_buf[m] = X->ndims[m];
		    }

		    /// For non-sorted Y
		    for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
		        ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
		    }

		    free(mode_order_X);
		    free(mode_order_Y);

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
		#pragma omp parallel for schedule(static) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds)
		    for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X.len - 1; ++fx_ptr) {    // Loop fiber pointers of X
		        int tid = omp_get_thread_num();
		        fx_counter--;
		        //if (fx_counter % 100 == 0) printf("Progress: %d\/%d\n", fx_counter, fidx_X.len);
		        if (tid == 0){
		            sptStartTimer(timer_SPA);
		        }
		        sptNnzIndex fx_begin = fidx_X.data[fx_ptr];
		        sptNnzIndex fx_end = fidx_X.data[fx_ptr+1];

		        /// hashtable size
		        const unsigned int ht_size = 10000;
		        sptIndex nmodes_spa = nmodes_Y - num_cmodes;
		        long int nnz_counter = 0;
		        sptIndex current_idx = 0;

		        table_t *ht;
		        ht = htCreate(ht_size);

		        if (tid == 0){
		            sptStopTimer(timer_SPA);
		            time_prep += sptElapsedTime(timer_SPA);
		        }

		        for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {
		            sptValue valX = X->values.data[zX];
		            if (tid == 0) {
		                sptStartTimer(timer_SPA);
		            }
		            sptIndexVector cmode_index_X;
		            sptNewIndexVector(&cmode_index_X, num_cmodes, num_cmodes);
		            for(sptIndex i = 0; i < num_cmodes; ++i){
		                cmode_index_X.data[i] = X->inds[nmodes_X - num_cmodes + i].data[zX];
		                //printf("\ncmode_index_X[%lu]: %lu\n", i, cmode_index_X.data[i]);
		            }

		            unsigned long long key_cmodes = 0;
		            for(sptIndex m = 0; m < num_cmodes; ++m)
		                key_cmodes += cmode_index_X.data[m] * Y_cmode_inds[m + 1];

		            tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
		            //printf("Y_val.len: %d\n", Y_val.len);
		            unsigned int my_len = Y_val.len;
		            if (tid == 0){
		                sptStopTimer(timer_SPA);
		                time_free_mode += sptElapsedTime(timer_SPA);
		            }
		            if(my_len == 0) continue;

		            if (tid == 0) {
		                sptStartTimer(timer_SPA);
		            }

		            for(int i = 0; i < my_len; i++){
		                unsigned long long fmode =  Y_val.key_FM[i];
		                //printf("i: %d, Y_val.key_FM[i]: %lu, Y_val.val[i]: %f\n", i, Y_val.key_FM[i], Y_val.val[i]);
		                sptValue spa_val = htGet(ht, fmode);
		                float result = Y_val.val[i] * valX;
		                if(spa_val == LONG_MIN) {
		                    htInsert(ht, fmode, result);
		                    nnz_counter++;
		                }
		                else
		                    htUpdate(ht, fmode, spa_val + result);
		            }

		            if (tid == 0){
		                sptStopTimer(timer_SPA);
		                time_spa += sptElapsedTime(timer_SPA);
		            }

		        }   // End Loop nnzs inside a X fiber

		        if (tid == 0) {
		            sptStartTimer(timer_SPA);
		        }

		        for(int i = 0; i < ht->size; i++){
		            node_t *temp = ht->list[i];
		            while(temp){
		                unsigned long long idx_tmp = temp->key;
		                //nnz_counter++;
		                for(sptIndex m = 0; m < nmodes_spa; ++m) {
		                    //printf("idx_tmp: %lu, m: %d, (idx_tmp inds_buf[m])/inds_buf[m+1]): %d\n", idx_tmp, m, (idx_tmp%inds_buf[m])/inds_buf[m+1]);
		                    sptAppendIndexVector(&Z_tmp[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
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
		        htFree(ht);
		        if (tid == 0){
		            sptStopTimer(timer_SPA);
		            time_accumulate_z += sptElapsedTime(timer_SPA);
		        }
		    }

		sptStopTimer(timer);
		double main_computation = sptElapsedTime(timer);
		total_time += main_computation;
		double spa_total = time_prep + time_free_mode + time_spa + time_accumulate_z;
		printf("[Index Search]: %.6f s\n", (time_free_mode + time_prep)/spa_total * main_computation);
		printf("[Accumulation]: %.6f s\n", (time_spa + time_accumulate_z)/spa_total * main_computation);

		sptStartTimer(timer);
		/// Append Z_tmp to Z
		    //Calculate the indecies of Z
		    unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
		    unsigned long long Z_total_size = 0;

		    Z_tmp_start[0] = 0;
		    for(int i = 0; i < tk; i++){
		        Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
		        Z_total_size +=  Z_tmp[i].nnz;
		        //printf("Z_tmp_start[i + 1]: %lu, i: %d\n", Z_tmp_start[i + 1], i);
		    }
		    //printf("%d\n", Z_total_size);
		    result = sptNewSparseTensorWithSize(Z, nmodes_Z, ndims_buf, Z_total_size);

		#pragma omp parallel for schedule(static) num_threads(tk) shared(Z, nmodes_Z, Z_tmp_start)
		    for(int i = 0; i < tk; i++){
		        int tid = omp_get_thread_num();
		        if(Z_tmp[tid].nnz > 0){
		            for(sptIndex m = 0; m < nmodes_Z; ++m)
		                sptAppendIndexVectorWithVectorStartFromNuma(&Z->inds[m], &Z_tmp[tid].inds[m], Z_tmp_start[tid]);
		            sptAppendValueVectorWithVectorStartFromNuma(&Z->values, &Z_tmp[tid].values, Z_tmp_start[tid]);
		            //sptDumpSparseTensor(&Z_tmp[tid], 0, stdout);
		        }
		    }

		    //  for(int i = 0; i < tk; i++)
		    //      sptFreeSparseTensor(&Z_tmp[i]);
		    sptStopTimer(timer);
		    total_time += sptPrintElapsedTime(timer, "Writeback");

		    sptStartTimer(timer);
		    if(output_sorting == 1){
		        sptSparseTensorSortIndex(Z, 1, tk);
		    }
		    sptStopTimer(timer);
		    total_time += sptPrintElapsedTime(timer, "Output Sorting");
		    printf("[Total time]: %.6f s\n", total_time);
		    printf("\n");
		}
}
