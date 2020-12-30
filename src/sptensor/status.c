/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <HiParTI.h>
#include "sptensor.h"
#include <math.h>


double ptiSparseTensorDensity(ptiSparseTensor const * const tsr)
{
  double root = pow((double)tsr->nnz, 1./(double)tsr->nmodes);
  double density = 1.0;
  for(ptiIndex m=0; m < tsr->nmodes; ++m) {
    density *= root / (double)tsr->ndims[m];
  }

  return density;
}



void ptiSparseTensorStatus(ptiSparseTensor *tsr, FILE *fp)
{
  fprintf(fp, "COO Sparse Tensor information (use ptiIndex, ptiValue))---------\n");
  fprintf(fp, "DIMS=%"HIPARTI_PRI_INDEX, tsr->ndims[0]);
  for(ptiIndex m=1; m < tsr->nmodes; ++m) {
    fprintf(fp, "x%"HIPARTI_PRI_INDEX, tsr->ndims[m]);
  }
  fprintf(fp, " NNZ=%"HIPARTI_PRI_NNZ_INDEX, tsr->nnz);
  fprintf(fp, " DENSITY=%e\n" , ptiSparseTensorDensity(tsr));

  fprintf(fp, "Average slice length (c): ");
  for(ptiIndex m=0; m < tsr->nmodes - 1; ++m) {
    fprintf(fp, "%.2lf , ", (double)tsr->nnz / tsr->ndims[m]);
  }
  fprintf(fp, "%.2lf\n", (double)tsr->nnz / tsr->ndims[tsr->nmodes-1]);

  char * bytestr = ptiBytesString(tsr->nnz * (sizeof(ptiIndex) * tsr->nmodes + sizeof(ptiValue)));
  fprintf(fp, "COO-STORAGE=%s\n", bytestr);
  fprintf(fp, "\n");
  free(bytestr);
}
