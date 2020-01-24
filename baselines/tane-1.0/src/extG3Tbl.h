#ifndef _H_EXTG3TBL
#define _H_EXTG3TBL	1

#include "extPartition.h"

void extG3TblINITIALIZE(int nooftuples);
void extG3TblLOAD(extPartition *partition);
void extG3TblUNLOAD(extPartition *partition);
int G3(extPartition *partition);
int extG3BiggestSet(extPartition *partition);


#endif
