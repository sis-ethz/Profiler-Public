#ifndef _H_EXTINPUTOUTPUT
#define _H_EXTINPUTOUTPUT	1

#include "extPartition.h"

void extInputOutputBEGIN(int nooftuples);
void extInputOutputEND(void);
extPartition *ReadPartition(char *filename);

#endif
