#include <ParTI.h>
#include <stdlib.h>
#include "sptensor.h"
#include <string.h>
#include <limits.h>
#include <numa.h>

/** All combined:
 * 0: COOY + SPA
 * 1: COOY + HTA
 * 2: HTY + SPA
 * 3: HTY + HTA
 * 4: HTY + HTA on HM
 **/
int sptSparseTensorMulTensor(sptSparseTensor * Z, sptSparseTensor * const X, sptSparseTensor *const Y, sptIndex num_cmodes, sptIndex * cmodes_X, sptIndex * cmodes_Y, int tk, int output_sorting, int placement)
{
	//	Experiment modes
	int experiment_modes;
	sscanf(getenv("EXPERIMENT_MODES"), "%d", &experiment_modes);

	//	Setup timer
	double total_time = 0;
	sptTimer timer;
	sptNewTimer(&timer, 0);

	//	Check inputs
	if(num_cmodes >= X->nmodes) {
		spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
	}
	for(sptIndex m = 0; m < num_cmodes; ++m) {
		if(X->ndims[cmodes_X[m]] != Y->ndims[cmodes_Y[m]]) {
			spt_CheckError(SPTERR_SHAPE_MISMATCH, "CPU  SpTns * SpTns", "shape mismatch");
		}
	}

	//	Initialize variables
	sptIndex nmodes_X= X->nmodes;
	sptIndex nmodes_Y= Y->nmodes;
	sptIndex nmodes_Z= nmodes_X + nmodes_Y - 2 * num_cmodes;

	sptNnzIndexVector fidx_X;
	sptNnzIndexVector fidx_Y;					// CooY 0.1
	table_t* Y_ht= tensor_htCreate(Y->nnz);		// HtY	2.3.4

	sptSparseTensor* Z_tmp= (sptSparseTensor*)malloc(tk * sizeof (sptSparseTensor));
	sptIndex* ndims_buf= (sptIndex*)malloc(nmodes_Z * sizeof(sptIndex));

	sptIndex* Y_cmode_inds= (sptIndex*)malloc((num_cmodes + 1) * sizeof(sptIndex));
	sptIndex* Y_fmode_inds= (sptIndex*)malloc((nmodes_Y - num_cmodes + 1) * sizeof(sptIndex));

	//	Start Experiment
		//	0: COOY + SPA
	if(experiment_modes == 0){
		sptStartTimer(timer);
			process_X(X, nmodes_X, num_cmodes, cmodes_X, tk, &fidx_X);
			process_CooY(Y, nmodes_Y, num_cmodes, cmodes_Y, tk, &fidx_Y);
			prepare_Z(X, Y, num_cmodes, nmodes_X, nmodes_Y, nmodes_Z, tk, ndims_buf, Z_tmp, cmodes_Y);
		sptStopTimer(timer);
		total_time += sptElapsedTime(timer);
		printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer));

		sptStartTimer(timer);
			compute_CooY_SpZ(&fidx_X, &fidx_Y, nmodes_X, nmodes_Y, num_cmodes, tk, Z_tmp, X, Y);
			//combine_Z(Z, nmodes_Z, tk, ndims_buf, Z_tmp);
		sptStopTimer(timer);
		total_time += sptElapsedTime(timer);
		printf("[Computation]: %.6f s\n", sptElapsedTime(timer));

		sptStartTimer(timer);
			sptSparseTensorSortIndex(Z, 1, tk);
		sptStopTimer(timer);
		total_time += sptElapsedTime(timer);
		printf("[Output Processing]: %.6f s\n", sptElapsedTime(timer));

		printf("[Total time]: %.6f s\n", total_time);
	}

		//	3: HTY + HTA
	if(experiment_modes == 3){
		sptStartTimer(timer);
			process_X(X, nmodes_X, num_cmodes, cmodes_X, tk, &fidx_X);
			process_HtY(Y, nmodes_Y, num_cmodes, cmodes_Y, tk, Y_ht, Y_cmode_inds, Y_fmode_inds);
			prepare_Z(X, Y, num_cmodes, nmodes_X, nmodes_Y, nmodes_Z, tk, ndims_buf, Z_tmp, cmodes_Y);
		sptStopTimer(timer);
		total_time += sptElapsedTime(timer);
		printf("[Input Processing]: %.6f s\n", sptElapsedTime(timer));

		sptStartTimer(timer);
			compute_HtY_HtZ(&fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds, Z_tmp, tk, X);
			combine_Z(Z, nmodes_Z, tk, ndims_buf, Z_tmp);
		sptStopTimer(timer);
		total_time += sptElapsedTime(timer);
		printf("[Computation]: %.6f s\n", sptElapsedTime(timer));

		sptStartTimer(timer);
			sptSparseTensorSortIndex(Z, 1, tk);
		sptStopTimer(timer);
		total_time += sptElapsedTime(timer);
		printf("[Output Processing]: %.6f s\n", sptElapsedTime(timer));

		printf("[Total time]: %.6f s\n", total_time);
	}

	return 0;
}

/**
 * Find mode_order for X,Y
 */
void find_mode(sptIndex * mode_order, sptIndex * cmodes, sptIndex nmodes, sptIndex num_cmodes,
		sptIndex ci, sptIndex fi)
{
	//	find free modes
	for(sptIndex m = 0; m < nmodes; ++m) {
		if(sptInArray(cmodes, num_cmodes, m) == -1) {
			mode_order[fi] = m;
			++ fi;
		}
	}
	//	find contract modes
	for(sptIndex m = 0; m < num_cmodes; ++m) {
		mode_order[ci] = cmodes[m];
		++ ci;
	}

	return;
}

/**
 * Sort X to be free--contract
 */
void process_X(sptSparseTensor * const X, sptIndex nmodes_X, sptIndex num_cmodes,
		sptIndex * cmodes_X, int tk, sptNnzIndexVector * fidx_X)
{
	//	find mode_order
	sptIndex* mode_order_X = (sptIndex *)malloc(nmodes_X * sizeof(sptIndex));
	sptIndex ci = nmodes_X - num_cmodes;
	sptIndex fi = 0;
	find_mode(mode_order_X, cmodes_X, nmodes_X, num_cmodes, ci, fi);

	//	sort X w.r.t. mode_order_X
	sptSparseTensorShuffleModes(X, mode_order_X);
	sptSparseTensorSortIndex(X, 1, tk);
	for(sptIndex m = 0; m < nmodes_X; ++m)	mode_order_X[m] = m;

	// set fidx_X as indices for outer loop
	sptSparseTensorSetIndices(X, mode_order_X, nmodes_X - num_cmodes, fidx_X);

	free(mode_order_X);
	return;
}

/**
 * Sort Y to be contract--free
 */
void process_CooY(sptSparseTensor * const Y, sptIndex nmodes_Y, sptIndex num_cmodes,
		sptIndex * cmodes_Y, int tk, sptNnzIndexVector * fidx_Y)
{
	//	find mode_order
	sptIndex* mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
	sptIndex ci = 0;
	sptIndex fi = num_cmodes;
	find_mode(mode_order_Y, cmodes_Y, nmodes_Y, num_cmodes, ci, fi);

	//	sort Y w.r.t. mode_order_Y
	sptSparseTensorShuffleModes(Y, mode_order_Y);
	sptSparseTensorSortIndex(Y, 1, tk);
	for(sptIndex m = 0; m < nmodes_Y; ++m)	mode_order_Y[m] = m;

	//	set fidx_Y as indices for inner loop
	sptSparseTensorSetIndices(Y, mode_order_Y, num_cmodes, fidx_Y);

	free(mode_order_Y);
	return;
}

/**
 * Convert Y to be hash-table
 */
void process_HtY(sptSparseTensor * const Y, sptIndex nmodes_Y, sptIndex num_cmodes,
		sptIndex * cmodes_Y, int tk,
		table_t * Y_ht, sptIndex * Y_cmode_inds, sptIndex * Y_fmode_inds)
{
	//	find mode order
	sptIndex* mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
	sptIndex ci = 0;
	sptIndex fi = num_cmodes;
	find_mode(mode_order_Y, cmodes_Y, nmodes_Y, num_cmodes, ci, fi);

	//	calculate key range for Y hashtable
	for(sptIndex i = 0; i < num_cmodes + 1; i++)
		Y_cmode_inds[i] = 1;
	for(sptIndex i = 0; i < num_cmodes; i++){
		for(sptIndex j = i; j < num_cmodes; j++)
			Y_cmode_inds[i] = Y_cmode_inds[i] * Y->ndims[mode_order_Y[j]];
	}

	sptIndex Y_num_fmodes = nmodes_Y - num_cmodes;
	for(sptIndex i = 0; i < Y_num_fmodes + 1; i++)
		Y_fmode_inds[i] = 1;
	for(sptIndex i = 0; i < Y_num_fmodes; i++){
		for(sptIndex j = i; j < Y_num_fmodes; j++)
			Y_fmode_inds[i] = Y_fmode_inds[i] * Y->ndims[mode_order_Y[j + num_cmodes]];
	}

	unsigned int Y_ht_size = Y->nnz;
	sptNnzIndex Y_nnz = Y->nnz;

	//	create omp lock
	omp_lock_t *locks = (omp_lock_t *)malloc(Y_ht_size*sizeof(omp_lock_t));
	for(size_t i = 0; i < Y_ht_size; i++) omp_init_lock(&locks[i]);

	#pragma omp parallel for schedule(static) num_threads(tk) shared(Y_ht, Y_num_fmodes, mode_order_Y, num_cmodes, Y_cmode_inds, Y_fmode_inds)
		for(sptNnzIndex i = 0; i < Y_nnz; i++){ // parallel on each non_zero

			//	find hash_table key for the non_zero
			unsigned long long key_cmodes = 0;
			for(sptIndex m = 0; m < num_cmodes; ++m)
				key_cmodes += Y->inds[mode_order_Y[m]].data[i] * Y_cmode_inds[m + 1];

			unsigned long long key_fmodes = 0;
			for(sptIndex m = 0; m < Y_num_fmodes; ++m)
				key_fmodes += Y->inds[mode_order_Y[m+num_cmodes]].data[i] * Y_fmode_inds[m + 1];

			//	insert the non_zero into hash_table
			unsigned pos = tensor_htHashCode(key_cmodes);
			omp_set_lock(&locks[pos]);

			tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
			if(Y_val.len == 0) {
				tensor_htInsert(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
			}
			else  {
				tensor_htUpdate(Y_ht, key_cmodes, key_fmodes, Y->values.data[i]);
			}

			omp_unset_lock(&locks[pos]);
		}

	//	discard omp lock
	for(size_t i = 0; i < Y_ht_size; i++) omp_destroy_lock(&locks[i]);

	free(mode_order_Y);
	free(locks);
	return;
}


/**
 * Reserve memory of Z for threads
 */
void prepare_Z(sptSparseTensor * const X, sptSparseTensor * const Y,
		sptIndex num_cmodes, sptIndex nmodes_X, sptIndex nmodes_Y, sptIndex nmodes_Z,
		int tk, sptIndex * ndims_buf, sptSparseTensor * Z_tmp, sptIndex * cmodes_Y)
{
	//	find Y mode order
	sptIndex* mode_order_Y = (sptIndex *)malloc(nmodes_Y * sizeof(sptIndex));
	sptIndex ci = 0, fi = num_cmodes;
	find_mode(mode_order_Y, cmodes_Y, nmodes_Y, num_cmodes, ci, fi);

	//	allocate modes dimensions for Z
	for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
		ndims_buf[m] = X->ndims[m];
	}
	for(sptIndex m = num_cmodes; m < nmodes_Y; ++m) {
		ndims_buf[(m - num_cmodes) + nmodes_X - num_cmodes] = Y->ndims[mode_order_Y[m]];
	}
	free(mode_order_Y);

	int result;
	//	allocate a local Z_tmp for each thread
	for (int i = 0; i < tk; i++){
		result = sptNewSparseTensor(&(Z_tmp[i]), nmodes_Z, ndims_buf);
	}

	spt_CheckError(result, "CPU  SpTns * SpTns", NULL);

	free(mode_order_Y);
	return;
}

/**
 * Computation via CooFormat-Y and SparseAccumulator-Z
 */
void compute_CooY_SpZ(sptNnzIndexVector * fidx_X, sptNnzIndexVector * fidx_Y, sptIndex nmodes_X,
		sptIndex nmodes_Y, sptIndex num_cmodes, int tk, sptSparseTensor * Z_tmp, sptSparseTensor * const X, sptSparseTensor * const Y)
{
#pragma omp parallel for schedule(static) num_threads(tk) shared(fidx_X, fidx_Y, nmodes_X, nmodes_Y, num_cmodes, Z_tmp)
	for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X->len - 1; ++fx_ptr) { // parallel on X-fibers
		int tid = omp_get_thread_num();

		sptNnzIndex fx_begin = fidx_X->data[fx_ptr];
		sptNnzIndex fx_end = fidx_X->data[fx_ptr+1];

		//	allocate buffers to read from CooY
		sptIndex nmodes_spa = nmodes_Y - num_cmodes;

		sptIndexVector * spa_inds = (sptIndexVector*)malloc(nmodes_spa * sizeof(sptIndexVector));
		sptValueVector spa_vals;
		for(sptIndex m = 0; m < nmodes_spa; ++m)
			sptNewIndexVector(&spa_inds[m], 0, 0);
		sptNewValueVector(&spa_vals, 0, 0);

		sptIndexVector inds_buf;
		sptNewIndexVector(&inds_buf, (nmodes_Y - num_cmodes), (nmodes_Y - num_cmodes));

		//	(non-parallel) loop through non_zeroes in the X-fiber
		for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {

			//	find value and index of the X-non_zero
			sptValue valX = X->values.data[zX];
			sptIndexVector cmode_index_X;
			sptNewIndexVector(&cmode_index_X, num_cmodes, num_cmodes);
			for(sptIndex i = 0; i < num_cmodes; ++i){
				 cmode_index_X.data[i] = X->inds[nmodes_X - num_cmodes + i].data[zX];
			 }

			sptNnzIndex fy_begin = -1;
			sptNnzIndex fy_end = -1;

			//	find the corresponding Y-fiber
			for(sptIndex j = 0; j < fidx_Y->len; j++){
				for(sptIndex i = 0; i< num_cmodes; i++){
					if(cmode_index_X.data[i] != Y->inds[i].data[fidx_Y->data[j]])
						break;
					if(i == (num_cmodes - 1)){
						fy_begin = fidx_Y->data[j];
						fy_end = fidx_Y->data[j+1];
						break;
					}
				}
				if (fy_begin != -1 || fy_end != -1)
					break;
			}

			//	if no Y-fiber is found, skip to next X-non_zero
			if (fy_begin == -1 || fy_end == -1)
				continue;

			char tmp[32];
			char index_str[128];
			long int tmp_key;

			//	(non-parallel) loop through non_zeroes in the Y-fiber
			for(sptNnzIndex zY = fy_begin; zY < fy_end; ++ zY) {
				for(sptIndex m = 0; m < nmodes_spa; ++m)
					inds_buf.data[m] = Y->inds[m + num_cmodes].data[zY];
				long int found = sptInIndexVector(spa_inds, nmodes_spa, spa_inds[0].len, &inds_buf);
				if( found == -1) {
					for(sptIndex m = 0; m < nmodes_spa; ++m)
						sptAppendIndexVector(&spa_inds[m], Y->inds[m + num_cmodes].data[zY]);
					sptAppendValueVector(&spa_vals, Y->values.data[zY] * valX);
				} else {
					spa_vals.data[found] += Y->values.data[zY] * valX;
				}
			}

		}

		//	write to local Z_tmp
		Z_tmp[tid].nnz += spa_vals.len;

		for(sptIndex i = 0; i < spa_vals.len; ++i) {
			for(sptIndex m = 0; m < nmodes_X - num_cmodes; ++m) {
				sptAppendIndexVector(&Z_tmp[tid].inds[m], X->inds[m].data[fx_begin]);
			}
		}
		for(sptIndex m = 0; m < nmodes_spa; ++m)
			sptAppendIndexVectorWithVector(&Z_tmp[tid].inds[m + (nmodes_X - num_cmodes)], &spa_inds[m]);
		sptAppendValueVectorWithVector(&Z_tmp[tid].values, &spa_vals);

		//	free buffers
		for(sptIndex m = 0; m < nmodes_spa; ++m){
			sptFreeIndexVector(&(spa_inds[m]));
		 }
		 sptFreeValueVector(&spa_vals);
	}

	return;
}

/**
 * Computation via HashTable-Y and HashTable-Z
 */
void compute_HtY_HtZ(sptNnzIndexVector * fidx_X, sptIndex nmodes_X, sptIndex nmodes_Y, sptIndex num_cmodes,
		sptIndex * Y_fmode_inds, table_t * Y_ht, sptIndex * Y_cmode_inds, sptSparseTensor * Z_tmp, int tk, sptSparseTensor * const X)
{
#pragma omp parallel for schedule(static) num_threads(tk) shared(fidx_X, nmodes_X, nmodes_Y, num_cmodes, Y_fmode_inds, Y_ht, Y_cmode_inds, Z_tmp)
	for(sptNnzIndex fx_ptr = 0; fx_ptr < fidx_X->len - 1; ++fx_ptr) { // parallel on X-fibers
		int tid = omp_get_thread_num();

		sptNnzIndex fx_begin = fidx_X->data[fx_ptr];
		sptNnzIndex fx_end = fidx_X->data[fx_ptr+1];

		//	allocate hashtable to store intermediate result
		const unsigned int ht_size = 10000;
		table_t *ht;
		ht = htCreate(ht_size);

		sptIndex nmodes_spa = nmodes_Y - num_cmodes;
		long int nnz_counter = 0;
		sptIndex current_idx = 0;

		//	(non-parallel) loop through non_zeroes in the X-fiber
		for(sptNnzIndex zX = fx_begin; zX < fx_end; ++ zX) {

			//	find value and index of the X-non_zero
			sptValue valX = X->values.data[zX];
			sptIndexVector cmode_index_X;
			sptNewIndexVector(&cmode_index_X, num_cmodes, num_cmodes);
			for(sptIndex i = 0; i < num_cmodes; ++i){
				cmode_index_X.data[i] = X->inds[nmodes_X - num_cmodes + i].data[zX];
			}

			//	calculate key for HtY
			unsigned long long key_cmodes = 0;
			for(sptIndex m = 0; m < num_cmodes; ++m)
				key_cmodes += cmode_index_X.data[m] * Y_cmode_inds[m + 1];

			tensor_value Y_val = tensor_htGet(Y_ht, key_cmodes);
			unsigned int my_len = Y_val.len;

			if(my_len == 0) continue;

			//	(non-parallel) loop through non_zeroes in the HtY entry
			for(int i = 0; i < my_len; i++){
				unsigned long long fmode =  Y_val.key_FM[i];
				sptValue spa_val = htGet(ht, fmode);
				float result = Y_val.val[i] * valX;
				if(spa_val == LONG_MIN) {
					htInsert(ht, fmode, result);
					nnz_counter++;
				}
				else
					htUpdate(ht, fmode, spa_val + result);
			}
		}

		//	write to local Z_tmp
		for(int i = 0; i < ht->size; i++){
			node_t *temp = ht->list[i];
			while(temp){
				unsigned long long idx_tmp = temp->key;
				for(sptIndex m = 0; m < nmodes_spa; ++m) {
					sptAppendIndexVector(&Z_tmp[tid].inds[m + (nmodes_X - num_cmodes)], (idx_tmp%Y_fmode_inds[m])/Y_fmode_inds[m+1]);
				}
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
	}

	return;
}

/**
 * Combine Z-tmp's to Z
 */
void combine_Z(sptSparseTensor * Z, sptIndex nmodes_Z, int tk, sptIndex * ndims_buf, sptSparseTensor * Z_tmp)
{
	//	calculate total number of indices
	unsigned long long* Z_tmp_start = (unsigned long long*) malloc( (tk + 1) * sizeof(unsigned long long));
	unsigned long long Z_total_size = 0;

	Z_tmp_start[0] = 0;
	for(int i = 0; i < tk; i++){
		Z_tmp_start[i + 1] = Z_tmp[i].nnz + Z_tmp_start[i];
		Z_total_size +=  Z_tmp[i].nnz;
	}
	int result = sptNewSparseTensorWithSize(Z, nmodes_Z, *ndims_buf, Z_total_size);

#pragma omp parallel for schedule(static) num_threads(tk) shared(Z, nmodes_Z, Z_tmp_start)
	for(int i = 0; i < tk; i++){ // parallel on each Z-tmp
		int tid = omp_get_thread_num();

		//	insert to Z if contain non-zero
		if(Z_tmp[tid].nnz > 0){
			for(sptIndex m = 0; m < nmodes_Z; ++m)
				sptAppendIndexVectorWithVectorStartFromNuma(&Z->inds[m], &Z_tmp[tid].inds[m], Z_tmp_start[tid]);
			sptAppendValueVectorWithVectorStartFromNuma(&Z->values, &Z_tmp[tid].values, Z_tmp_start[tid]);
		}
	}

	return;
}
