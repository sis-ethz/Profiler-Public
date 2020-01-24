#ifndef _H_EXTINTERSECTIONTBL
#define _H_EXTINTERSECTIONTBL	1

#include "extPartition.h"
#ifdef USEDISK
#include "extPartitionFile.h"
#endif

void extIntersectionTblINITIALIZE(int nooftuples);
void extIntersectionTblLOAD(extPartition *partition);
void extIntersectionTblUNLOAD(extPartition *partition);
#ifndef USEDISK
extPartition *Intersection(extPartition *partition);
#else
extPartition *Intersection(extPartition *partition,extPartitionFile *file);
#endif

extPartition *InitialPartition(int *table, int nooftuples);

#endif
